import os
import pprint
import re
import numpy as np
import pandas as pd
import xlrd
import json
import openpyxl
from sklearn.model_selection import train_test_split

global_annotations = [0]


def read_high_recall_matcher_output(high_recall_matcher_output):
    post_df = pd.read_csv(high_recall_matcher_output)
    return post_df


def read_annotations_data(annotations_data_path):
    post_df = pd.read_csv(annotations_data_path)
    return post_df


def get_associations_idx(loc_hrm, loc_ann_list):
    """Returns the closest offset and the absolute distance from the tagged terms (in the corresponding post)
    from the manual annotations"""

    distance = np.abs(loc_hrm - np.array(loc_ann_list))
    return loc_ann_list[np.argmin(distance)], np.min(distance)


def get_windows_and_labels_from_data(hrm_df_data, annotations_data, window_size):
    windows = []
    hrm_umls = []
    annotations_umls = []
    labels = []

    annotations_data = annotations_data.to_dict()
    posts_dict = get_posts_dict(annotations_data)
    posts_seen = {}

    # for each entry in the data (post), we go over the matches found
    for i in hrm_df_data['matches_found'].index:
        entry = hrm_df_data.loc[i]
        post_txt = entry['post_txt']
        tokenized_txt = entry['tokenized_text']

        # avoid going over duplicate posts
        if tokenized_txt not in list(posts_seen.keys()):
            posts_seen[tokenized_txt] = 1

            match_entry = hrm_df_data['matches_found'][i]
            match_entry_json = json.loads(match_entry)

            # each match json contains many terms. using the offset of the term we create a window around
            # each term to get the context of it. We then collect the UMLS tagged by the HRM output, and the UMLS
            # chosen by the manual annotators or Nan if there was no corresponding UMLS tagged manually
            # (i.e. - the comparison between the 2 UMLS will be the final label).
            for cand_match in match_entry_json:
                # (1) context window around the term
                candidate = cand_match['cand_match']
                original_offset = cand_match['curr_occurence_offset']
                word_offset = get_match_word_offset(original_offset, post_txt)
                word_count = len(candidate.split())
                window = get_window_for_candidate(post_txt=post_txt,
                                                  start_word_offset=word_offset,
                                                  end_word_offset=word_offset + word_count - 1,
                                                  window_size=window_size,
                                                  pad=False)
                windows.append(window)

                # (2) UMLS from high recall matcher
                umls = cand_match['umls_match']
                hrm_umls.append(umls)

                # (3) UMLS (or NAN) from manual annotations
                umls2 = None
                if tokenized_txt in list(posts_dict.keys()):
                    closest_offset, distance = get_associations_idx(original_offset,
                                                                    list(posts_dict[tokenized_txt].keys()))
                    if distance < 10:
                        # since there is a gap between the terms from the HRM and the ones from the annotations data,
                        # we use the offset to find the specific match from the annotations data and then collect the
                        # corresponding UMLS
                        umls2, post_num, annotation_id = posts_dict[tokenized_txt][closest_offset]

                annotations_umls.append(umls2)

                # (4) ground-truth label (comparison of the HRM UMLS to the manual annotations UMLS)
                labels.append(1 if umls == umls2 or compare_expressions(candidate, umls2) else 0)

                # (5) final step: expand HRM results by adding terms that have no CUI
                # i.e. - don't have a CUI
                expansion_sizes = [1, 2]
                for expansion_size in expansion_sizes:
                    expanded_term = expand_term(post_txt=post_txt,
                                                start_word_offset=word_offset,
                                                end_word_offset=word_offset + word_count - 1,
                                                expansion_size=expansion_size)
                    expanded_term_window = get_window_for_candidate(post_txt=post_txt,
                                                                    start_word_offset=word_offset,
                                                                    end_word_offset=word_offset + expansion_size,
                                                                    window_size=window_size,
                                                                    pad=False)

                    windows.append(expanded_term_window)
                    hrm_umls.append(expanded_term)
                    annotations_umls.append(umls2)
                    labels.append(1 if expanded_term == umls2 or compare_expressions(expanded_term, umls2) else 0)

    return {'windows': windows,
            'HRM_UMLS': hrm_umls,
            'Annotations_UMLS': annotations_umls,
            'Labels': labels}


def compare_expressions(exp1, exp2):
    """compares 2 string expressions and returns True if the two differ in at most the first character for any one
    of the expression's words.
    e.g. - חולה הסכרת vs. לחולה סכרת will return True"""
    if exp1 is None or exp2 is None:
        return False

    exp1 = str(exp1).lower()
    exp2 = str(exp2).lower()

    exp1_words = exp1.split()
    exp2_words = exp2.split()

    if len(exp1_words) == len(exp2_words):
        ans = True
        for i, word in enumerate(exp1_words):
            word2 = exp2_words[i]
            ans = ans and (word == word2 or word[1:] == word2 or word == word2[1:])
        return ans
    else:
        return False


def fix_manual_annotations(data, hrm_cui, window, umls, annotations_data, umls_data, cand_match):
    umls2, post_num, annotation_id = data
    annotations_cui = get_cui_from_word(umls2, umls_data)

    # if there is no CUI for the annotations' match, then we want to choose the HRM CUI instead, meaning label as 1 (HRM 'got it right')
    if annotations_cui is None and not compare_expressions(cand_match, umls2):
        global_annotations[0] += 1
        print(str(post_num))
        print(str(window) +
              '\n' +
              'hrm_cand_match: ' +
              str(cand_match) +
              ', ' +
              str(hrm_cui) +
              ', ' +
              str(umls) +
              ', annotations UMLS that was not found: ' +
              str(umls2))
        # 1 if the HRM got it right, otherwise 0
        val = 0  # input("is recall matcher right: ")

        if val == '1':
            # we update the annotations
            annotations_data['merged_inner_and_outer'][post_num][annotation_id]['term'] = str(umls)
            # save to file
            with open('output_data/corrected_annotations_temp.json', 'w', encoding='utf-8') as data_file:
                json.dump(annotations_data, data_file, ensure_ascii=False, indent=4)
            print('saved to file...')

        print('\n\n')


def print_windows(windows, hrm_umls, annotations_umls):
    for i, item in enumerate(windows):
        print('window ' + str(i) + ': ' + str(item) +
              '\n' +
              'High recall matcher UMLS: ' + str(hrm_umls[i]) +
              '\n' +
              'Annotations UMLS: ' +
              str(annotations_umls[i]) + '\n')
    print('training data size: ' + str(len(windows)))


def get_posts_dict(annotations_data):
    """Returns a dictionary from each post to a dictionary that maps the offset of the matches found in the post
    to the terms themselves"""

    posts_dict = {}

    for post_num in range(len(annotations_data['merged_inner_and_outer'])):
        post_txt = annotations_data['tokenized_text'][post_num]
        if type(annotations_data['merged_inner_and_outer'][post_num]) is str:
            annotations_data['merged_inner_and_outer'][post_num] = \
                json.loads(annotations_data['merged_inner_and_outer'][post_num])
        offset_to_term_dict = {}
        for ann_id, term_json in enumerate(annotations_data['merged_inner_and_outer'][post_num]):
            offset_to_term_dict[term_json['start_offset']] = [term_json['term'], post_num, ann_id]
        posts_dict[post_txt] = offset_to_term_dict

    return posts_dict


def get_cui_from_word(candidate, umls_data):
    results_idx = [i for i, x in enumerate(umls_data['HEB']) if candidate == x]
    return umls_data.loc[results_idx[0]]['CUI'] if len(results_idx) else None


def get_word_from_offset(sentence, offset):
    ans = ''
    while 0 <= offset < len(sentence) and sentence[offset] != ' ':
        ans += sentence[offset]
        offset += 1
    return ans


def get_match_word_offset(char_offset, row):
    """Returns the word offset based on the input char offset"""
    word_offset = 0
    while char_offset > 0:
        if row[char_offset] == ' ':
            word_offset += 1
            while row[char_offset] == ' ':
                char_offset -= 1
        else:
            char_offset -= 1
    return word_offset


def expand_term(post_txt, start_word_offset, end_word_offset, expansion_size=1, pad=False):
    post_txt = post_txt.split(' ')
    post_txt = [w for w in post_txt if len(w)]
    match_phrase = get_phrase_from_text_by_offsets(post_txt, start_word_offset, end_word_offset)
    ans = [match_phrase]

    i = 1
    while expansion_size > 0:
        # add one word after the match word
        if 0 <= end_word_offset + i < len(post_txt):
            ans.insert(len(ans), post_txt[end_word_offset + i])
        # if doesn't exist - pad
        elif pad:
            ans.insert(len(ans), '*')
        i += 1
        expansion_size -= 1

    return ' '.join(ans)


def get_phrase_from_text_by_offsets(text, start_word_offset, end_word_offset):
    word_count = end_word_offset - start_word_offset + 1
    match_phrase = []

    i = 0
    while i < word_count:
        if 0 <= start_word_offset + i < len(text):
            match_phrase.append(text[start_word_offset + i])
        i += 1

    return ' '.join(match_phrase)


def get_window_for_candidate(post_txt, start_word_offset, end_word_offset, window_size=3, pad=False):
    post_txt = post_txt.split(' ')
    post_txt = [w for w in post_txt if len(w)]
    match_phrase = get_phrase_from_text_by_offsets(post_txt, start_word_offset, end_word_offset)
    ans = [match_phrase]

    i = 1
    while window_size > 0:
        # take one word before the match phrase
        if 0 <= start_word_offset - i < len(post_txt):
            ans.insert(0, post_txt[start_word_offset - i])
        # if doesn't exist - pad
        elif pad:
            ans.insert(0, '*')

        # take one word after the match phrase
        if 0 <= end_word_offset + i < len(post_txt):
            ans.insert(len(ans), post_txt[end_word_offset + i])
        # if doesn't exist - pad
        elif pad:
            ans.insert(len(ans), '*')
        i += 1
        window_size -= 1

    # returns:
    # ['שלום', 'אני', 'חולה', 'סוכרת', 'ורציתי']
    # as a sentence where 'חולה' is the match word
    return ' '.join(ans)


def google_counts(query):
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"}
    URL = 'https://www.google.com/search?q="{}"'.format(query)
    result = requests.get(URL, headers=headers)

    soup = BeautifulSoup(result.content, 'html.parser')
    print(result.text)
    total_results_text = soup.find("div", {"id": "result-stats"}).find(text=True,
                                                                       recursive=False)  # this will give you the outer text which is like 'About 1,410,000,000 results'
    results_num = ''.join([num for num in total_results_text if
                           num.isdigit()])  # now will clean it up and remove all the characters that are not a number .
    return int(results_num)


def main():
    # data paths
    data_dir = r"C:\Users\Rins\Desktop\data" + os.sep
    annotations_data_path = data_dir + r'manual_labeled_v2\doccano\merged_output\diabetes_labels.csv'
    # annotations_data_path = 'output_data/corrected_annotations.json'
    high_recall_matcher__path = data_dir + r'high_recall_matcher\output\diabetes.csv'

    # data sources
    high_recall_matcher_output = read_high_recall_matcher_output(high_recall_matcher__path)
    annotations_data = read_annotations_data(annotations_data_path)

    window_size = 2
    train, test = train_test_split(high_recall_matcher_output, test_size=0.1)
    train = get_windows_and_labels_from_data(train, annotations_data, window_size)
    test = get_windows_and_labels_from_data(test, annotations_data, window_size)

    data = {'train': train, 'test': test}
    # save the data to file
    with open('output_data/training_data_{}.json'.format(window_size), 'w', encoding='utf-8') as data_file:
        json.dump(data,
                  data_file,
                  ensure_ascii=False,
                  indent=4)


if __name__ == '__main__':
    main()
