import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from config import COMMUNITIES
from training_data.utils import get_posts_dict, read_csv_data, get_span_to_offset_dict, save_to_file, \
    read_dict_from_json_file

community_name = "diabetes"
data_dir = r"C:\Users\Rins\Desktop\data" + os.sep
hrm_data_path = r'high_recall_matcher\heb_to_eng_mrconso_disorders_chemicals_kb.csv'
eng_to_heb_file_path = data_dir + hrm_data_path.format(community_name)
eng_to_heb_file = pd.read_csv(eng_to_heb_file_path)
heb_to_cui_dict_file_path = 'json_files/heb_to_cui_dict.json'


def compute_span_umls_score(spans_vectors_char, umls_vectors_char, spans_vectors_word, umls_vectors_word, i):
    k_w = 0.2

    sim_vector_char = cosine_similarity(spans_vectors_char[i], umls_vectors_char)[0]
    sim_vector_word = cosine_similarity(spans_vectors_word[i], umls_vectors_word)[0]

    score = sim_vector_char + k_w*sim_vector_word
    normalized_score = score / (1 + k_w)

    return sim_vector_char#normalized_score


def create_heb_to_cui_dict():
    heb_to_cui_dict = {}
    for i in range(eng_to_heb_file.shape[0]):
        heb_name = eng_to_heb_file.loc[i]['HEB']
        cui = eng_to_heb_file.loc[i]['CUI']

        heb_to_cui_dict[heb_name] = cui

    save_to_file(heb_to_cui_dict, is_hrm=True, is_constant=True, name='heb_to_cui_dict')


def main():
    num_of_best_matches = 2
    sim_threshold = 0.4

    #create_heb_to_cui_dict()
    span_sizes = [0, 1, 2]  # how many words we want to append to current phrase (span)
    heb_to_cui_dict_from_file = read_dict_from_json_file(heb_to_cui_dict_file_path)

    # from gensim.sklearn_api import W2VTransformer
    # vectorizer = W2VTransformer(size=10, min_count=1, seed=1)

    # bigram char analyzer
    # source: https://mjeensung.github.io/characterbigramtfidf/
    char_vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='char', ngram_range=(2, 2))
    word_vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', ngram_range=(1, 3))

    spans = []
    for community in COMMUNITIES:
        # manual annotations
        annotations_data_path = data_dir + r'manual_labeled_v2\doccano\merged_output\{}_labels.csv'.format(community)
        annotations_data = read_csv_data(annotations_data_path)

        # hrm output data
        high_recall_matcher_path = data_dir + r'high_recall_matcher\output\{}.csv'.format(community)
        high_recall_matcher_output = read_csv_data(high_recall_matcher_path)

        hrm_df_data, _ = train_test_split(high_recall_matcher_output, test_size=0.1)
        annotations_data_dict = annotations_data.to_dict()
        posts_dict, _, _ = get_posts_dict(annotations_data_dict)

        span_data = {}
        # go over the posts
        annotations_data_dict['post_txt'] = {}
        for i in annotations_data_dict['tokenized_text'].keys():
            tokenized_text = annotations_data_dict['tokenized_text'][i]
            post_nums = [k for k, t in high_recall_matcher_output['tokenized_text'].items() if t == tokenized_text][0]
            #original_text = annotations_data_dict['text'][i]
            original_text = high_recall_matcher_output['post_txt'][post_nums]

            annotations_data_dict['post_txt'][i] = original_text
            span_data[original_text] = {}
            for span_size in span_sizes:
                span_to_offset_dict = get_span_to_offset_dict(original_text, span_size)
                span_data[original_text].update(span_to_offset_dict)
                # collect the spans
                spans += list(span_to_offset_dict.keys())

        terms_umls, terms_spans = [], []

        # spans
        for post_span in spans:
            #post_span = re.sub('[!@#$,.?;"]', '', post_span).strip()
            if post_span:
                terms_spans.append(post_span)

        # heb UMLS
        for heb in heb_to_cui_dict_from_file.keys():
            terms_umls += [heb]

        # combine to one list
        terms = terms_spans + terms_umls

        # tansformer
        # terms_train = [[t] for t in terms]
        # vectorizer.fit(terms_train)

        # spans_vectors = []
        # for t in terms_train[:len(terms_spans)]:
        #     spans_vectors.append(vectorizer.transform(t))
        # spans_vectors = np.array(spans_vectors)
        #
        # umls_vectors = []
        # for t in terms_train[len(terms_spans):]:
        #     umls_vectors.append(vectorizer.transform(t)[0])
        # umls_vectors = np.array(umls_vectors)

        #TF IDF

        tfidf_vectorizer_vectors_char = char_vectorizer.fit_transform(terms)
        tfidf_vectorizer_vectors_word = word_vectorizer.fit_transform(terms)

        spans_vectors_char = tfidf_vectorizer_vectors_char[:len(terms_spans)]
        umls_vectors_char = tfidf_vectorizer_vectors_char[len(terms_spans):]

        spans_vectors_word = tfidf_vectorizer_vectors_word[:len(terms_spans)]
        umls_vectors_word = tfidf_vectorizer_vectors_word[len(terms_spans):]

        terms_umls = np.array(terms_umls)
        spans_to_vector_id = dict(zip(terms_spans, range(len(terms_spans))))
        annotations_data_dict['matches_found'] = {}
        for original_text, span_to_offset_dict in span_data.items():
            # in case there are duplicate posts
            post_nums = [k for k, t in annotations_data_dict['post_txt'].items() if t == original_text]
            for p in post_nums:
                annotations_data_dict['matches_found'][p] = []

            for span, offset in span_to_offset_dict.items():
                i = spans_to_vector_id[span]
                # compute score for the span (mention) and all UMLS
                score = compute_span_umls_score(spans_vectors_char, umls_vectors_char, spans_vectors_word, umls_vectors_word, i)
                if np.sum(score):  # check if there exists at least one positive sim value
                    idx = np.argsort(score)[::-1][:num_of_best_matches]  # take <num_of_best_matches> max sim values
                    idx = idx[score[idx] > sim_threshold]  # take indices of the above values iff they pass the defined threshold
                    if len(idx):
                        for p in post_nums:
                            for umls_match in terms_umls[idx]:
                                entry = {'cand_match': span, 'umls_match': umls_match, 'curr_occurence_offset': offset}
                                annotations_data_dict['matches_found'][p].append(entry)

        save_to_file(data=annotations_data_dict, is_constant=True, is_hrm=True, name='our_hrm_sim_th_0.4_bigram/our_hrm_{}'.format(community))


if __name__ == '__main__':
    main()
