import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import COMMUNITIES, data_dir, WINDOW_SIZES
from training_data.training_data_obj import TrainingData, TrainingDataExtended
from training_data.utils import save_to_file, read_csv_data, expand_term, compare_expressions, get_word_offset, \
    get_window_for_candidate, get_posts_dict, get_closest_offset, read_dict_from_json_file


def get_windows_and_labels_from_data(hrm_df_data, annotations_data, window_size):
    heb_to_cui_dict_file_path = 'json_files/heb_to_cui_dict.json'
    heb_to_cui_dict_from_file = read_dict_from_json_file(heb_to_cui_dict_file_path)

    baseline_training_data = TrainingData()
    extended_training_data = TrainingDataExtended()
    ixd = [i for i, t in enumerate(hrm_df_data['tokenized_text'].tolist()) if t in annotations_data['tokenized_text'].tolist()]
    hrm_df_data = hrm_df_data.iloc[ixd]
    annotations_data_dict = annotations_data.to_dict()
    posts_dict, annotations_count, cui_only_annotations_count = get_posts_dict(annotations_data_dict)

    annotations_matched = defaultdict(int)  # i.e., HRM's recall
    annotations_matched_cui_only = defaultdict(int)  # i.e., HRM's recall for CUI only
    posts_seen = {}
    garbage_windows_max = 1000
    sum_pro = 0
    # for each entry in the data (post), we go over the matches found
    for i in hrm_df_data['matches_found'].index:
        entry = hrm_df_data.loc[i]
        tokenized_txt, match_entry, post_txt = entry['tokenized_text'], entry['matches_found'], entry['post_txt']

        if type(match_entry) == str:
            match_entry = json.loads(match_entry)

        # avoid going over duplicate posts
        if post_txt not in list(posts_seen.keys()):
            posts_seen[post_txt] = post_txt
            sum_pro += len(match_entry)
            # each match json contains many terms. using the offset of the term we create a window around
            # each term to get the context of it. We then collect the UMLS tagged by the HRM output, and the UMLS
            # chosen by the manual annotators or Nan if there was no corresponding UMLS tagged manually
            # (i.e. - the comparison between the 2 UMLS will be the final label).
            for cand_match in match_entry:
                # (1) context window around the term
                candidate, char_offset = cand_match['cand_match'], cand_match['curr_occurence_offset']
                word_offset = get_word_offset(char_offset, post_txt)
                word_count = len(candidate.split())
                window = get_window_for_candidate(post_txt=post_txt,
                                                  start_word_offset=word_offset,
                                                  end_word_offset=word_offset + word_count - 1,
                                                  window_size=window_size,
                                                  pad=True)

                # (2) UMLS from high recall matcher
                umls = cand_match['umls_match']

                # (3) UMLS (or NAN) from manual annotations
                umls2 = None

                # if tokenized_txt in list(posts_dict.keys()):
                closest_offset, distance = get_closest_offset(char_offset,
                                                              list(posts_dict[tokenized_txt].keys()))
                # since there is a gap between the terms from the HRM and the ones from the annotations data,
                # we use the offset to find the specific match from the annotations data and then collect the
                # corresponding UMLS
                if distance < 15: # TODO: remove?
                    offset_to_term_dict = posts_dict[tokenized_txt]
                    umls2, post_num, annotation_id = offset_to_term_dict[closest_offset]

                # (4) ground-truth label (comparison of the HRM UMLS to the manual annotations UMLS)
                label = 1 if umls == umls2 or compare_expressions(candidate, umls2) else 0
                cui_only_label = label and umls2 in heb_to_cui_dict_from_file

                # making sure we don't match each annotation more than one HRM candidate
                annotations_matched[(post_txt, closest_offset)] = max(annotations_matched[(post_txt, closest_offset)], label)
                annotations_matched_cui_only[(post_txt, closest_offset)] = max(annotations_matched_cui_only[(post_txt, closest_offset)], cui_only_label)

                # baseline output is collected up until step 5, without performing the expansion
                baseline_training_data.add_data(window=window, hrm_umls=umls, annotations_umls=umls2, label=label)
                extended_training_data.add_data(window=window, hrm_umls=umls, annotations_umls=umls2, label=label, is_expanded=0)

                # (5) final step: expand HRM results by adding terms that have no CUI
                # (5.1) we try to expand the umls itself by adding 1 or 2 words to it, up to a 3 word length
                expansion_sizes = range(3 - word_count)
                for expansion_size in expansion_sizes:
                    expanded_term = expand_term(post_txt=post_txt,
                                                start_word_offset=word_offset,
                                                end_word_offset=word_offset + word_count - 1,
                                                expansion_size=expansion_size + 1)

                    expansion_label = 1 if expanded_term == umls2 or compare_expressions(expanded_term, umls2) else 0
                    annotations_matched[(post_txt, closest_offset)] = max(annotations_matched[(post_txt, closest_offset)], expansion_label)

                    # (5.3) if the expanded term matches the annotations umls then we add it to the result
                    if expansion_label == 1:
                        # we use the original window and simply change the term and label
                        extended_training_data.add_data(window=window, hrm_umls=expanded_term, annotations_umls=umls2,
                                                        label=expansion_label, is_expanded=1)

                    # otherwise, we want to monitor the number of 'garbage' windows we add to the result data so that
                    # it doesn't outweigh the number of successful expansions
                    elif garbage_windows_max:
                        extended_training_data.add_data(window=window, hrm_umls=expanded_term, annotations_umls=umls2,
                                                        label=expansion_label, is_expanded=1)
                        garbage_windows_max -= 1

    print('Total matches found', sum_pro)
    print('Recall: ', str(sum(annotations_matched.values())) + '/' + str(annotations_count) + ' = ' + str(sum(annotations_matched.values()) / annotations_count))
    print('Recall from CUI only: ', sum(annotations_matched_cui_only.values()) / cui_only_annotations_count)
    return baseline_training_data.get(), extended_training_data.get(), annotations_count


def main():
    np.random.seed(3)
    for community in COMMUNITIES:
        # manual annotations
        annotations_data_path = data_dir + r'manual_labeled_v2\doccano\merged_output\{}_labels.csv'.format(community)
        annotations_data = read_csv_data(annotations_data_path)

        # MDTEL's hrm output data
        high_recall_matcher_path = data_dir + r'high_recall_matcher\output\{}.csv'.format(community)
        high_recall_matcher_output = read_csv_data(high_recall_matcher_path)

        # Our hrm output data
        our_high_recall_matcher_output_path = 'json_files/hrm/our_hrm_sim_th_0.4_bigram/our_hrm_{}.json'.format(community)
        with open(our_high_recall_matcher_output_path, encoding="utf8") as json_file:
            our_high_recall_matcher_output = pd.DataFrame.from_dict(json.load(json_file))  # dict

        train_data, test_data = train_test_split(annotations_data, test_size=0.1)

        # construct train and test sets for each window, using the split above as input
        for window_size in WINDOW_SIZES:
            print('\n\n')
            print(str(community), str(window_size))

            baseline_training_data, extended_training_data, training_annotations_count = get_windows_and_labels_from_data(
                our_high_recall_matcher_output,
                train_data,
                window_size)

            baseline_testing_data, extended_testing_data, testing_annotations_count = get_windows_and_labels_from_data(
                our_high_recall_matcher_output,
                test_data,
                window_size)

            data = {'baseline': {'train': baseline_training_data, 'test': baseline_testing_data},
                    'data': {'train': extended_training_data, 'test': extended_testing_data}}

            print('----------- BASELINE MODEL -----------')
            print('# of UMLS (positive labels) in training data: ' + str(np.sum(baseline_training_data['Labels'])))
            print('# of UMLS (positive labels) in testing data: ' + str(np.sum(baseline_testing_data['Labels'])))
            print('# of total training labels: ' + str(len(baseline_training_data['Labels'])))
            print('# of total testing labels: ' + str(len(baseline_testing_data['Labels'])))

            print('----------- EXPENDED MODEL -----------')
            print('# of UMLS (positive labels) in training data: ' + str(np.sum(extended_training_data['Labels'])))
            print('# of UMLS (positive labels) in testing data: ' + str(np.sum(extended_testing_data['Labels'])))
            print('# of total training labels: ' + str(len(extended_training_data['Labels'])))
            print('# of total testing labels: ' + str(len(extended_testing_data['Labels'])))

            print('--------------------------------------')
            print('Total annotations in expended train set: ', training_annotations_count)
            print('Total annotations in expended test set: ', testing_annotations_count)

            # save the output to file
            #save_to_file(data, community, window_size)

        break


if __name__ == '__main__':
    main()
