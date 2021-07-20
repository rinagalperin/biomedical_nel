import pickle
import pandas as pd
import numpy as np
import progressbar

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from training_data.utils import read_csv_data, get_span_to_offset_list, \
    read_dict_from_json_file, get_heb_stop_words, clean_text, save_data_to_file


def create_heb_to_cui_dict(eng_to_heb_file_path):
    eng_to_heb_file = pd.read_csv(eng_to_heb_file_path)

    heb_to_cui_dict = {}
    for i in range(eng_to_heb_file.shape[0]):
        heb_name = eng_to_heb_file.loc[i]['HEB']
        cui = eng_to_heb_file.loc[i]['CUI']

        heb_to_cui_dict[heb_name] = cui

    save_data_to_file(heb_to_cui_dict, file_name='heb_to_cui_dict', folder_name='')


def enrich_umls(train_data, eng_umls):
    changed = 0
    lang_to_cui_dict_file_path = 'json_files/hrm/eng_to_cui_dict.json'
    umls_to_cui = read_dict_from_json_file(lang_to_cui_dict_file_path)
    to_add = []

    for doc_num in train_data['file_name']:
        annotations = train_data['merged_inner_and_outer'][doc_num]
        text = train_data['text'][doc_num]
        for annotation in annotations:
            term = annotation['term']
            a, b = int(annotation['start_offset']), int(annotation['end_offset'])
            mention = text[a:b]
            clean_mention = clean_text(mention)
            # if no word in term is in the annotated mention, then we want to enrich our umls list with it
            if not len([word for word in clean_mention.split() if word in term]):
                if clean_mention not in umls_to_cui and len(clean_mention) > 1:
                    to_add.append(clean_mention)
                    umls_to_cui[clean_mention] = umls_to_cui[term]
                    changed += 1
    eng_umls = np.concatenate((eng_umls, to_add))
    save_data_to_file(data=umls_to_cui, file_name='eng_to_cui_dict_enriched.json', folder_name='json_files/hrm')
    print(changed, ' total enriched umls')
    return eng_umls


def get_eng_data(eng_umls, span_sizes, stop_words, annotations_data_file_path, split=False):
    spans = []

    annotations_data_dict = read_dict_from_json_file(annotations_data_file_path)
    if split:
        data, enrichment_data = train_test_split(pd.DataFrame.from_dict(annotations_data_dict), test_size=0.1)
        _, data_sample = train_test_split(data, test_size=0.1)

        # enrichment
        eng_umls = enrich_umls(enrichment_data, eng_umls)

        annotations_data_dict = pd.DataFrame.to_dict(data_sample)
    else:
        annotations_data_dict = pd.DataFrame.from_dict(annotations_data_dict).to_dict()

    span_data = {}
    # go over the posts
    annotations_data_dict['post_txt'] = {}
    for id, original_text in annotations_data_dict['text'].items():
        annotations_data_dict['post_txt'][id] = original_text
        span_data[original_text] = []
        for span_size in span_sizes:
            span_to_offset_list = get_span_to_offset_list(original_text, span_size, stop_words)
            span_data[original_text].extend(span_to_offset_list)
            # collect the spans
            spans.extend(np.array(span_to_offset_list)[:, 0].tolist())

    return list(set(spans)), span_data, annotations_data_dict, eng_umls


def get_heb_data(community, span_sizes, stop_words):
    spans = []
    # manual annotations
    annotations_data_path = 'C:/Users/Rins/Desktop/data/manual_labeled_v2/doccano/merged_output/{}_labels.csv'.format(community)
    annotations_data = read_csv_data(annotations_data_path)

    # hrm output data
    high_recall_matcher_path = 'C:/Users/Rins/Desktop/data/high_recall_matcher/output/{}.csv'.format(community)
    high_recall_matcher_output = read_csv_data(high_recall_matcher_path)

    annotations_data_dict = annotations_data.to_dict()
    # _, test_data = train_test_split(pd.DataFrame.from_dict(annotations_data_dict), test_size=0.1)
    # annotations_data_dict = pd.DataFrame.to_dict(test_data)

    span_data = {}
    annotations_data_dict['post_txt'] = {}
    for i in annotations_data_dict['tokenized_text'].keys():
        tokenized_text = annotations_data_dict['tokenized_text'][i]
        post_nums = [k for k, t in high_recall_matcher_output['tokenized_text'].items() if t == tokenized_text][0]
        original_text = high_recall_matcher_output['post_txt'][post_nums]

        annotations_data_dict['post_txt'][i] = original_text
        span_data[original_text] = []
        for span_size in span_sizes:
            span_to_offset_list = get_span_to_offset_list(original_text, span_size, stop_words)
            span_data[original_text].extend(span_to_offset_list)
            # collect the spans
            spans.extend(np.array(span_to_offset_list)[:, 0].tolist())

    terms_spans = []

    # spans
    for post_span in spans:
        if post_span:
            terms_spans.append(post_span)

    return list(set(terms_spans)), span_data, annotations_data_dict


def main_eng(annotations_data_file_path, eng_umls, file_name, folder_name):
    from nltk.corpus import stopwords
    eng_stop_words = set(stopwords.words('english'))
    span_sizes = [0, 1, 2]  # range of words we want to append to current phrase (span)
    terms_spans, span_data, annotations_data_dict, eng_umls = get_eng_data(eng_umls=eng_umls,
                                                                                       span_sizes=span_sizes,
                                                                                       stop_words=eng_stop_words,
                                                                                       annotations_data_file_path=annotations_data_file_path,
                                                                                       split=True)
    run_hrm(terms_umls=eng_umls, terms_spans=np.array(terms_spans), span_data=span_data,
            annotations_data_dict=annotations_data_dict, file_name=file_name, folder_name=folder_name)


def main_heb(heb_to_cui_dict_file_path, heb_file_name, heb_folder_name, community):
    heb_umls = []

    heb_stop_words = get_heb_stop_words()

    span_sizes = [0, 1, 2]  # how many words we want to append to current phrase (span)
    terms_spans, span_data, annotations_data_dict = get_heb_data(community=community,
                                                                 span_sizes=span_sizes,
                                                                 stop_words=heb_stop_words)

    heb_to_cui_dict_from_file = read_dict_from_json_file(heb_to_cui_dict_file_path)
    for heb in heb_to_cui_dict_from_file.keys():
        heb_umls += [heb]

    run_hrm(terms_umls=np.array(heb_umls), terms_spans=np.array(terms_spans), span_data=span_data,
            annotations_data_dict=annotations_data_dict, file_name=heb_file_name, folder_name=heb_folder_name)


def train_tf_idf_vectorizer(terms_spans, terms_umls):
    # bigram char analyzer
    # source: https://mjeensung.github.io/characterbigramtfidf/
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))

    terms = np.concatenate((terms_spans, terms_umls))
    tfidf_vectorizer_vectors_char = char_vectorizer.fit_transform(terms)
    umls_vectors_char = tfidf_vectorizer_vectors_char[len(terms_spans):]
    spans_vectors_char = tfidf_vectorizer_vectors_char[:len(terms_spans)]

    return umls_vectors_char, spans_vectors_char


def run_hrm(terms_umls, terms_spans, span_data, annotations_data_dict, file_name, folder_name):
    print('Starting HRM...')
    num_of_best_matches = 50
    sim_threshold = 0.4
    umls_vectors_char, spans_vectors_char = train_tf_idf_vectorizer(terms_spans, terms_umls)
    spans_to_vector_id = dict(zip(terms_spans, range(len(terms_spans))))
    annotations_data_dict['matches_found'] = {}
    spans_per_post = list(span_data.items())
    for i in progressbar.progressbar(range(len(spans_per_post))):
        original_text, span_to_offset_list = spans_per_post[i]
        spans = np.array(span_to_offset_list)[:, 0]
        # in case there are duplicate posts
        post_nums = [k for k, t in annotations_data_dict['post_txt'].items() if t == original_text]
        # compute score for the span (mention) and all UMLS
        spans_idx = np.array([spans_to_vector_id[span] for span in spans])
        all_score = cosine_similarity(spans_vectors_char[spans_idx], umls_vectors_char)

        matches_found = []
        for i, (span, offset) in enumerate(span_to_offset_list):
            score = all_score[i]
            idx = np.argsort(score)[::-1][:num_of_best_matches]  # take <num_of_best_matches> max sim values
            idx = idx[score[idx] > sim_threshold]  # take indices of the above values iff they pass the defined th
            if len(idx):
                for umls_match in terms_umls[idx]:
                    entry = {'cand_match': span, 'umls_match': umls_match, 'curr_occurence_offset': offset}
                    matches_found.append(entry)

        for p in post_nums:
            annotations_data_dict['matches_found'][p] = []
            annotations_data_dict['matches_found'][p] = matches_found

    save_data_to_file(data=annotations_data_dict, file_name=file_name, folder_name=folder_name)


def main():
    heb_to_cui_dict_file_path = 'json_files/hrm/heb_to_cui_dict.json'
    heb_folder_name = 'E:/nlp_model/hrm/camoni'
    community = 'sclerosis'
    heb_file_name = 'heb_{}_tfidf_cosim'.format(community)

    medmentions_annotations_data_file_path = 'json_files/annotations_data/med_mentions_annotations_data'
    eng_umls_file_path = 'json_files/hrm/eng_umls.pkl'
    with open(eng_umls_file_path, 'rb') as f:
        eng_umls = pickle.load(f)
    medmentions_folder_name = 'E:/nlp_model/hrm/medmentions'
    medmentions_file_name = 'eng_1'

    bc5cdr_annotations_data_file_path = 'json_files/annotations_data/bc5cdr_annotations_data'
    bc5cdr_name_to_id_file_path = '../data_preprocessing/data/bc5cdr/mesh_name_to_id_dict.json'
    bc5cdr_name_to_id_dict = read_dict_from_json_file(bc5cdr_name_to_id_file_path)
    bc5cdr_names = np.array(list(bc5cdr_name_to_id_dict.keys()))
    bc5cdr_folder_name = 'E:/nlp_model/hrm/bc5cdr'
    bc5cdr_file_name = 'eng_enriched.json'

    np.random.seed(3)
    main_eng(medmentions_annotations_data_file_path, eng_umls, medmentions_file_name, medmentions_folder_name)
    #main_heb(heb_to_cui_dict_file_path, heb_file_name, heb_folder_name, community)


if __name__ == '__main__':
    main()
