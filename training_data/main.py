import os
import pprint
import re
import numpy as np
import pandas as pd
import xlrd
import json
import openpyxl


def read_high_recall_matcher_output(high_recall_matcher_output):
    post_df = pd.read_csv(high_recall_matcher_output)
    return post_df


def read_annotations_data(annotations_data_path):
    post_df = pd.read_json(annotations_data_path)
    return post_df


def get_windows_and_labels_from_data(hrm_df_data, umls_data, annotations_data, window_size):
    windows = []
    hrm_umls = []
    annotations_umls = []
    labels = []

    annotations_data = annotations_data.to_dict()
    posts_dict = get_posts_dict(annotations_data)
    #posts_seen = {}

    # for each entry in the data (post), we go over the matches found
    for i in range(len(hrm_df_data['matches_found'])):
        entry = hrm_df_data.iloc[i]
        post_txt = entry['post_txt']
        tokenized_txt = entry['tokenized_text']

        # avoid going over duplicate posts
        #if tokenized_txt not in list(posts_seen.keys()):
        #   posts_seen[tokenized_txt] = 1

        match_entry = hrm_df_data['matches_found'][i]
        match_entry_json = json.loads(match_entry)

        # each match json contains many terms. using the offset of the term we create a window around
        # each term to get the context of it. We then collect the CUI from the HRM output, and the CUI
        # that corresponds with the UMLS chosen by the manual annotators (i.e. - the ground truth label).
        for cand_match in match_entry_json:
            original_offset = cand_match['curr_occurence_offset']
            word_offset = get_match_word_offset(original_offset, post_txt)
            window = get_window_for_candidate(post_txt, word_offset, window_size, False)
            windows.append(window)

            # UMLS CUI from high recall matcher
            umls = cand_match['umls_match']
            hrm_umls.append(umls)


            # UMLS (or NAN) from manual annotations
            umls2 = None
            if tokenized_txt in list(posts_dict.keys()) and original_offset in list(posts_dict[tokenized_txt].keys()):
                # since there is a gap between the terms from the HRM and the ones from the annotations data,
                # we use the offset to find the specific match from the annotations data and then collect the
                # corresponding CUI
                umls2, post_num, annotation_id = posts_dict[tokenized_txt][original_offset]

                # annotations_cui = get_cui_from_word(umls2, umls_data)
                # hrm_cui = cand_match['cui']
                # if there is no CUI for the annotations' match, then we want to choose the HRM CUI instead, meaning label as 1 (HRM 'got it right')
                # if annotations_cui is None:
                #     print(str(window) + '\n' + 'hrm_cui: ' + str(hrm_cui) + ', ' + str(umls) + ', annotations UMLS that was not found: ' + str(umls2))
                #
                #     # 1 if the HRM got it right, otherwise 0
                #     val = input("is recall matcher right: ")
                #
                #     if val == '1':
                #         # we update the annotations
                #         annotations_data['merged_inner_and_outer'][post_num][annotation_id]['term'] = str(umls)
                #         # save to file
                #         with open('output_data/corrected_annotations.json', 'w', encoding='utf-8') as data_file:
                #             json.dump(annotations_data, data_file, ensure_ascii=False, indent=4)
                #         print('saved to file...')
                #
                #     print('\n\n')

            annotations_umls.append(umls2)
            labels.append(1 if umls == umls2 else 0)


    return windows, hrm_umls, annotations_umls, labels


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
        #annotations_data['merged_inner_and_outer'][post_num] = json.loads(annotations_data['merged_inner_and_outer'][post_num])
        offset_to_term_dict = {}

        for ann_id, term_json in enumerate(annotations_data['merged_inner_and_outer'][post_num]):
            offset_to_term_dict[term_json['start_offset']] = [term_json['term'], post_num, ann_id]
            # improves from 1859 Nans to 1516
            offset_to_term_dict[term_json['start_offset'] + 1] = [term_json['term'], post_num, ann_id]
            offset_to_term_dict[term_json['start_offset'] - 1] = [term_json['term'], post_num, ann_id]
            # improves from 1516 Nans to 1505 then stabilizes
            offset_to_term_dict[term_json['start_offset'] + 2] = [term_json['term'], post_num, ann_id]
            offset_to_term_dict[term_json['start_offset'] - 2] = [term_json['term'], post_num, ann_id]
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


def get_window_for_candidate(post_txt, word_offset, window_size=3, pad=False):
    post_txt = post_txt.split(' ')
    post_txt = [w for w in post_txt if len(w)]
    match_word = post_txt[word_offset]
    ans = [match_word]
    i = 1
    while window_size > 0:
        # take one word before the match word
        if 0 <= word_offset - i < len(post_txt):
            ans.insert(0, post_txt[word_offset - i])
        # if doesn't exist - pad
        elif pad:
            ans.insert(0, '*')

        # take one word after the match word
        if 0 <= word_offset + i < len(post_txt):
            ans.insert(len(ans), post_txt[word_offset + i])
        # if doesn't exist - pad
        elif pad:
            ans.insert(len(ans), '*')
        i += 1
        window_size -= 1

    # returns:
    # ['שלום', 'אני', 'חולה', 'סוכרת', 'ורציתי']
    # as a sentence where 'חולה' is the match word
    return ' '.join(ans)


def main():
    # data paths
    data_dir = r"C:\Users\Rins\Desktop\data" + os.sep
    umls_df_data_path = data_dir + r"high_recall_matcher\heb_to_eng_mrconso_disorders_chemicals_kb.csv"
    #annotations_data_path = data_dir + r'manual_labeled_v2\doccano\merged_output\diabetes_labels.csv'
    annotations_data_path = 'output_data/corrected_annotations.json'
    high_recall_matcher__path = data_dir + r'high_recall_matcher\output\diabetes.csv'

    # data sources
    high_recall_matcher_output = read_high_recall_matcher_output(high_recall_matcher__path)
    umls_data = pd.read_csv(umls_df_data_path)
    annotations_data = read_annotations_data(annotations_data_path)

    window_size = 4
    #print(get_cui_from_word('חיסון שפעת חזירים', umls_data))
    windows, umls, annotations_umls, labels = get_windows_and_labels_from_data(high_recall_matcher_output,
                                                                       umls_data,
                                                                       annotations_data,
                                                                       window_size)

    # save the data to file
    with open('output_data/training_data_{}.json'.format(window_size), 'w', encoding='utf-8') as data_file:
        json.dump({'windows': windows,
                   'HRM_UMLS': umls,
                   'Annotations_UMLS': annotations_umls,
                   'Labels': labels},
                  data_file,
                  ensure_ascii=False,
                  indent=4)


if __name__ == '__main__':
    main()
