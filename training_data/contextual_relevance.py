import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import heb_data_dir, WINDOW_SIZES
from training_data.training_data_obj import TrainingData
from training_data.utils import read_csv_data, expand_term, compare_expressions, get_word_offset, \
    get_window_for_candidate, get_posts_dict, get_closest_offset, read_dict_from_json_file, save_data_to_file, \
    get_window_for_heb_candidate, clean_text
from enum import Enum


class Data(Enum):
    MDTEL = 1
    CAMONI = 2
    MED_MENTIONS = 3
    BC5CDR = 4


def balance_data_samples(data, annotations_hist, ratio=2):  # ratio of x:1 of negative examples count compared to positive examples count
    positive_examples_data = TrainingData()
    for umls_ann, window, d_from_ann, len_candidate, candidate in annotations_hist.values():
        positive_examples_data.add_data(window, umls_ann, umls_ann, 1)
    positive_examples_data = positive_examples_data.get()
    data_res = {}
    #positives_mask = np.array(data['Labels']) == 1
    positives_count = len(annotations_hist)#np.sum(positives_mask)

    # select desired number of negatives and add to result
    negatives_idx = np.flatnonzero(np.array(data['Labels']) == 0)
    negative_selection = np.random.choice(negatives_idx, positives_count*ratio)

    # add result
    for k in data:
        data_res[k] = np.array(positive_examples_data[k]).tolist()
        # data_res[k] += np.array(data[k])[negative_selection].tolist()
        data_res[k] += [data[k][i] for i in negative_selection]

    idx_shuffle = np.arange(positives_count + positives_count*ratio)
    np.random.shuffle(idx_shuffle)
    for k in data:
        data_res[k] = np.array(data_res[k])[idx_shuffle].tolist()

    return data_res


def get_windows_and_labels_from_data(hrm_df_data, annotations_data, window_size, lang_to_cui_dict_file_path, augment=False):
    training_data = TrainingData()
    ixd = [i for i, t in enumerate(hrm_df_data['tokenized_text'].tolist()) if t in annotations_data['tokenized_text'].tolist()]
    hrm_df_data = hrm_df_data.iloc[ixd]
    annotations_data_dict = annotations_data.to_dict()
    lang_to_cui_dict = read_dict_from_json_file(lang_to_cui_dict_file_path)
    posts_dict, annotations_count, cui_only_annotations_count = get_posts_dict(annotations_data_dict, lang_to_cui_dict)

    annotations_hist = {}
    annotations_matched = defaultdict(int)  # HRM's recall
    annotations_matched_cui_only = defaultdict(int)  # HRM's recall for CUI only
    posts_seen = {}
    sum_pro = 0

    # for each entry in the data (post), we go over the matches found
    for i in hrm_df_data['matches_found'].index:
        entry = hrm_df_data.loc[i]
        tokenized_txt, match_entry, post_txt = entry['tokenized_text'], entry['matches_found'], entry['post_txt']

        if type(match_entry) == str:
            match_entry = json.loads(match_entry)

        # avoid going over duplicate posts (if exist)
        if post_txt not in list(posts_seen.keys()):
            posts_seen[post_txt] = post_txt
            sum_pro += len(match_entry)
            for cand_match in match_entry:
                # (1) context window around the term
                candidate, char_offset = cand_match['cand_match'], cand_match['curr_occurence_offset']
                word_offset = get_word_offset(char_offset, post_txt)
                word_count = len(candidate.split())
                # window = get_window_for_heb_candidate(post_txt,
                #                                       start_word_offset=word_offset,
                #                                       end_word_offset=word_offset + word_count - 1,
                #                                       window_size=window_size,
                #                                       pad=True)
                window = get_window_for_candidate(tokenized_txt=tokenized_txt,
                                                  candidate=candidate,
                                                  char_offset=char_offset,
                                                  window_size=window_size,
                                                  pad=True)

                # (2) UMLS from high recall matcher
                umls = cand_match['umls_match']

                # (3) UMLS (or NAN) from manual annotations
                umls2 = None
                if list(posts_dict[tokenized_txt].keys()):
                    # if tokenized_txt in list(posts_dict.keys()):
                    closest_offset, _ = get_closest_offset(char_offset,
                                                                  list(posts_dict[tokenized_txt].keys()))
                    # since there is a gap between the terms from the HRM and the ones from the annotations data,
                    # we use the offset to find the specific match from the annotations data and then collect the
                    # corresponding UMLS
                    offset_to_term_dict = posts_dict[tokenized_txt]
                    umls2, post_num, annotation_id = offset_to_term_dict[closest_offset]

                    # (4) ground-truth label (comparison of the HRM UMLS to the manual annotations UMLS)
                    hrm_candidate_cui = lang_to_cui_dict.get(umls, None)
                    try:
                        label = 1 if hrm_candidate_cui == lang_to_cui_dict[umls2] else 0
                    except:
                        # in case annotators used umls name that is not in the KB
                        label = 1 if compare_expressions(umls, umls2) else 0

                    if label:
                        d_from_ann = np.abs(char_offset - int(closest_offset))
                        annotations_hist[annotation_id] = annotations_hist.get(annotation_id, [umls2, window, d_from_ann, len(candidate), candidate])
                        u, w, d, e, _ = annotations_hist[annotation_id]
                        if d > d_from_ann:
                            annotations_hist[annotation_id] = [umls2, window, d_from_ann, len(candidate), candidate]

                        if d == np.abs(char_offset - char_offset + len(candidate)):
                            if len(candidate) < e:
                                annotations_hist[annotation_id] = [umls2, window, d_from_ann, len(candidate), candidate]

                    cui_only_label = label and umls2 in lang_to_cui_dict

                    # making sure we don't match each annotation more than one HRM candidate
                    annotations_matched[(post_txt, closest_offset)] = max(annotations_matched[(post_txt, closest_offset)], label)
                    annotations_matched_cui_only[(post_txt, closest_offset)] = max(annotations_matched_cui_only[(post_txt, closest_offset)], cui_only_label)

                    training_data.add_data(window=window, hrm_umls=umls, annotations_umls=umls2, label=label)

                    if augment:  # Hebrew annotations patch, not necessary otherwise
                        # (5) final step: expand HRM results by adding terms that have no CUI
                        # (5.1) we try to expand the umls itself by adding 1 or 2 words to it, up to a 3 word length
                        expansion_sizes = range(3 - word_count)
                        for expansion_size in expansion_sizes:
                            expanded_term = expand_term(post_txt=post_txt,
                                                        start_word_offset=word_offset,
                                                        end_word_offset=word_offset + word_count - 1,
                                                        expansion_size=expansion_size + 1)

                            expansion_label = 1 if expanded_term == umls2 or compare_expressions(expanded_term,
                                                                                                 umls2) else 0
                            annotations_matched[(post_txt, closest_offset)] = max(
                                annotations_matched[(post_txt, closest_offset)],
                                expansion_label)

                            # (5.3) add expanded term to the result (whether it matches the annotations umls or not)
                            training_data.add_data(window=window, hrm_umls=expanded_term, annotations_umls=umls2,
                                                            label=expansion_label)

    print('Total matches found', sum_pro)
    print('Recall: ', str(sum(annotations_matched.values())) + '/' + str(annotations_count) + ' = ' + str(sum(annotations_matched.values()) / annotations_count))
    print('Recall from CUI only: ', sum(annotations_matched_cui_only.values()) / cui_only_annotations_count)
    return training_data.get(), annotations_hist, annotations_count-sum(annotations_matched.values())


def main(data):
    window_size = 2
    hrm_output_path, lang_to_cui_dict_file_path, output_file_name, output_folder_name, annotations_data = '', '', '', '', {}

    if data is Data.MED_MENTIONS.value:
        medmentions_annotations_data_file_path = 'json_files/annotations_data/med_mentions_annotations_data'
        with open(medmentions_annotations_data_file_path, encoding="utf8") as medmentions_json_file:
            annotations_data = pd.DataFrame.from_dict(json.load(medmentions_json_file))  # dict
        # MedMentions hrm output data
        hrm_output_path = 'E:/nlp_model/hrm/medmentions/eng_1.json'
        lang_to_cui_dict_file_path = 'json_files/hrm/eng_to_cui_dict_enriched.json'
        output_folder_name = 'json_files/contextual_relevance/eng'
        output_file_name = 'medmentions_1'
    elif data is Data.CAMONI.value:
        community = 'diabetes'
        annotations_data_path = heb_data_dir + r'manual_labeled_v2\doccano\merged_output\{}_labels.csv'.format(community)
        annotations_data = read_csv_data(annotations_data_path)
        # Heb hrm output data
        hrm_output_path = 'json_files/hrm/camoni/heb_{}.json'.format(community)
        lang_to_cui_dict_file_path = 'json_files/hrm/heb_to_cui_dict.json'
        output_folder_name = 'json_files/contextual_relevance/heb'
        output_file_name = 'heb_{}'.format(community)
    elif data is Data.MDTEL.value:
        community = 'sclerosis'
        annotations_data_path = heb_data_dir + r'manual_labeled_v2\doccano\merged_output\{}_labels.csv'.format(community)
        annotations_data = read_csv_data(annotations_data_path)
        # MDTEL's hrm output data
        hrm_output_path = heb_data_dir + r'high_recall_matcher\output\{}.csv'.format(community)
        lang_to_cui_dict_file_path = 'json_files/hrm/heb_to_cui_dict.json'
        output_folder_name = 'json_files/contextual_relevance/mdtel'
        output_file_name = 'mdtel_{}'.format(community)
    elif data is Data.BC5CDR.value:
        bc5cdr_annotations_data_file_path = 'json_files/annotations_data/bc5cdr_annotations_data'
        with open(bc5cdr_annotations_data_file_path, encoding="utf8") as bc5cdr_json_file:
            annotations_data = pd.DataFrame.from_dict(json.load(bc5cdr_json_file))  # dict
        # BC5CDR hrm output data
        hrm_output_path = 'E:/nlp_model/hrm/bc5cdr/eng.json'
        lang_to_cui_dict_file_path = '../data_preprocessing/data/bc5cdr/mesh_name_to_id_dict.json'
        #lang_to_cui_dict_file_path = 'E:/nlp_model/hrm/bc5cdr/mesh_name_to_id_dict_enriched.json'
        output_folder_name = 'json_files/contextual_relevance/eng'
        output_file_name = 'bc5cdr'
    else:
        print('invalid dataset')

    if data is Data.MDTEL.value:
        hrm_output = read_csv_data(hrm_output_path)
    else:
        with open(hrm_output_path, encoding="utf8") as json_file:
            hrm_output = pd.DataFrame.from_dict(json.load(json_file))  # df
    if data not in [Data.CAMONI.value, Data.MDTEL.value]:
        annotations_data = annotations_data.iloc[hrm_output['file_name']]

    if data is Data.BC5CDR.value:
        # BC5CDR has its own task test set
        bc5cdr_annotations_data_file_path_test = 'json_files/annotations_data/bc5cdr_annotations_data_test'
        with open(bc5cdr_annotations_data_file_path_test, encoding="utf8") as bc5cdr_test_json_file:
            annotations_data_test = pd.DataFrame.from_dict(json.load(bc5cdr_test_json_file))  # dict
        train_data = annotations_data
        test_data = annotations_data_test
        hrm_output_test = 'E:/nlp_model/hrm/bc5cdr/eng_test.json'
        with open(hrm_output_test, encoding="utf8") as json_file:
            hrm_output_test = pd.DataFrame.from_dict(json.load(json_file))
    else:
        train_data, test_data = train_test_split(annotations_data, test_size=0.1)

    # construct train and test sets for each window, using the split above as input
    print('starting contextual relevance data curation...')
    print('window size: ', str(window_size))

    training_data, training_annotations_hist, _ = get_windows_and_labels_from_data(
        hrm_output,
        train_data,
        window_size,
        lang_to_cui_dict_file_path,
        augment=data in [Data.CAMONI.value, Data.MDTEL.value])  # patch for hebrew annotations

    # balance ratio of positive/negative examples in training data sets
    training_data = balance_data_samples(training_data, training_annotations_hist,
                                         ratio=1 if data is Data.MED_MENTIONS.value else 3)

    testing_data, testing_annotations_hist, false_negatives = get_windows_and_labels_from_data(
        hrm_output if data is not Data.BC5CDR.value else hrm_output_test,
        test_data,
        window_size,
        lang_to_cui_dict_file_path,
        augment=data in [Data.CAMONI.value, Data.MDTEL.value])  # patch for hebrew annotations

    testing_data = balance_data_samples(testing_data, testing_annotations_hist, ratio=2)

    data = {'train': training_data, 'test': testing_data, 'false_negatives_test': false_negatives}

    print('# of UMLS (positive labels) in training data: ' + str(np.sum(training_data['Labels'])))
    print('# of UMLS (positive labels) in testing data: ' + str(np.sum(testing_data['Labels'])))
    print('# of total training labels: ' + str(len(training_data['Labels'])))
    print('# of total testing labels: ' + str(len(testing_data['Labels'])))
    print('# of false negatives in test set: ', false_negatives)

    # save the output to file
    save_data_to_file(data, output_file_name, output_folder_name)


if __name__ == '__main__':
    np.random.seed(3)
    main(Data.MED_MENTIONS.value)
