import pandas as pd
import numpy as np
import json
from itertools import islice


def read_csv_data(annotations_data_path):
    post_df = pd.read_csv(annotations_data_path)
    return post_df


def read_dict_from_json_file(file_path):
    with open(file_path, encoding="utf8") as json_file:
        return json.load(json_file)


def compare_expressions(exp1, exp2):
    """
    compares 2 string expressions and returns 'True' if the two differ in at most the first character for any one
    of the expression's words, provided that the character is a functional character in Hebrew.
    e.g. - חולה הסכרת vs. לחולה סכרת --> will return True since the expressions differ only in the first functional
    letter 'ל'
    """

    functional_characters = get_functional_characters()

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
            ans = ans and (word == word2 or (word[1:] == word2 and word[0] in functional_characters) or (
                    word == word2[1:] and word2[0] in functional_characters))
        return ans
    # the expressions don't have the same number of words
    else:
        return False


def get_closest_offset(loc_hrm, loc_ann_list):
    """
    Returns the closest offset and the absolute distance from the tagged terms (in the corresponding post)
    from the manual annotations
    """

    distance = np.abs(loc_hrm - np.array(loc_ann_list))
    return loc_ann_list[np.argmin(distance)], np.min(distance)


def get_posts_dict(annotations_data):
    """
    Returns a dictionary from each post to a dictionary that maps the offset of the matches found in the post
    to the terms themselves, post number and annotation id.
    """
    annotations_count = 0

    cui_only_annotations_count = 0
    heb_to_cui_dict_file_path = 'json_files/heb_to_cui_dict.json'
    heb_to_cui_dict_from_file = read_dict_from_json_file(heb_to_cui_dict_file_path)

    posts_dict = {}
    posts_seen = {}

    for post_num in annotations_data['tokenized_text'].keys():
        post_txt = annotations_data['tokenized_text'][post_num]
        if type(annotations_data['merged_inner_and_outer'][post_num]) is str:
            annotations_data['merged_inner_and_outer'][post_num] = \
                json.loads(annotations_data['merged_inner_and_outer'][post_num])
        offset_to_term_dict = {}
        for ann_id, term_json in enumerate(annotations_data['merged_inner_and_outer'][post_num]):
            if post_txt not in posts_seen:
                annotations_count += 1
                if term_json['term'] in heb_to_cui_dict_from_file:
                    cui_only_annotations_count += 1
            offset_to_term_dict[term_json['start_offset']] = [term_json['term'], post_num, ann_id]
        if post_txt not in posts_seen:
            posts_seen[post_txt] = post_txt
            posts_dict[post_txt] = offset_to_term_dict

    return posts_dict, annotations_count, cui_only_annotations_count


def get_cui_from_word(candidate, umls_data):
    results_idx = [i for i, x in enumerate(umls_data['HEB']) if candidate == x]
    return umls_data.loc[results_idx[0]]['CUI'] if len(results_idx) else None


def get_word_offset(char_offset, row):
    """
    Returns the word offset based on the input char offset
    """
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


def get_window_for_candidate(post_txt, start_word_offset, end_word_offset, window_size=3, pad=False, two_sided=True):
    """
    :param post_txt: text from which the word offsets were taken
    :param start_word_offset: starting offset of the phrase in post_txt
    :param end_word_offset: ending offset of the phrase in post_txt
    :param window_size: total words to add on each side
    :param pad: do we want to pad the window if there are no words to add
    :param two_sided: do we want to add words from both sides (if True: we also add from the right of the phrase)
    :return: window as str. e.g., ['שלום', 'אני', 'חולה', 'סוכרת', 'ורציתי']
             as a string sentence where 'חולה' is the match word surrounded by its context
    """
    post_txt = post_txt.split(' ')
    post_txt = [w for w in post_txt if len(w)]
    num_of_words_to_add = window_size
    # we take the phrase from post_txt
    # TODO: we removed empty strings from post_txt so the offset might be wrong now
    match_phrase = get_phrase_from_text_by_offsets(post_txt, start_word_offset, end_word_offset)

    # base window contains the original phrase
    ans = [match_phrase]

    # TODO: window size isn't working properly - when window_size=2 we're only adding 1 word from each side
    i = 1
    while num_of_words_to_add > 0:
        if two_sided:
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
        num_of_words_to_add -= 1

    return ' '.join(ans)


def window(seq, n=5, by_words=True):
    """
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    by_words: True if we want to split by words, otherwise - we split by characters.
    """
    it = iter(seq.split(' ')) if by_words else iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def save_to_file(data, community=None, window_size=None, is_hrm=False, is_constant=False, name=None):
    if is_constant:
        with open('json_files{}/{}.json'.format('/hrm' if is_hrm else '/contextual_relevance', name),
                  'w', encoding='utf-8') as data_file:
            json.dump(data,
                      data_file,
                      ensure_ascii=False,
                      indent=4)
    else:
        with open('json_files{}/training_data/data_{}_{}{}.json'.format('/hrm' if is_hrm else '/contextual_relevance',
                                                                        community,
                                                                        window_size,
                                                                        '_hrm' if is_hrm else ''),
                  'w', encoding='utf-8') as data_file:
            json.dump(data,
                      data_file,
                      ensure_ascii=False,
                      indent=4)


def get_span(stripped_text_arr, word, word_offset, size):
    loc_word_in_stripped_text = [i for i, w in enumerate(stripped_text_arr) if w == word]
    closest_offset, _ = get_closest_offset(word_offset, loc_word_in_stripped_text)
    ans = [stripped_text_arr[i] for i in range(closest_offset, min(closest_offset + size + 1, len(stripped_text_arr)))]
    return ' '.join(ans)


def get_span_to_offset_dict(original_text, window_size):
    span_to_offset_dict = {}
    stop_words = get_stop_words()
    original_text_arr = original_text.split(' ')
    stripped_text_arr = list(filter(None, original_text_arr))  # remove spaces (empty strings)
    for word_offset, word in enumerate(original_text_arr):
        if word:
            span_candidate = get_span(stripped_text_arr, word, word_offset, window_size)
            span_candidate_array = span_candidate.split(' ')

            # filter out empty string, spans starting or ending in a stop word, and spans which are not in the desired length
            if (len(span_candidate) and not (
                    clean_expression_from_functional_characters(span_candidate_array[0]) in stop_words or
                    clean_expression_from_functional_characters(span_candidate_array[-1]) in stop_words or
                    span_candidate_array[0] in stop_words or
                    span_candidate_array[-1] in stop_words or
                    len(span_candidate_array) != window_size + 1)):
                # find char offset from tokenized text
                tokenized_text_char_offset = get_char_offset_from_word_offset(original_text, word_offset)

                # clean the candidate from functional characters (each word)
                # clean_span_candidate = clean_expression_from_functional_characters(span_candidate)

                span_to_offset_dict[span_candidate] = tokenized_text_char_offset
                # if len(clean_span_candidate):
                #     span_to_offset_dict[clean_span_candidate] = tokenized_text_char_offset

    return span_to_offset_dict


def get_char_offset_from_word_offset(text, word_offset):
    words_up_to_given_word_arr = text.split(' ')[:word_offset]
    words_up_to_given_word_str = ' '.join(words_up_to_given_word_arr)
    return len(words_up_to_given_word_str)


def clean_expression_from_functional_characters(expression):
    """
    remove functional characters from each word of the input expression
    """
    expression_array = expression.split(' ')
    functional_characters = get_functional_characters()
    for j, expression_array_member in enumerate(expression_array):
        if expression_array_member and expression_array_member[0] in functional_characters:
            expression_array[j] = expression_array_member[1:]
    return ' '.join(expression_array)


def get_stop_words():
    return [
        'אני',
        'את',
        'אתה',
        'אנחנו',
        'אתן',
        'אתם',
        'הם',
        'הן',
        'היא',
        'הוא',
        'שלי',
        'שלו',
        'שלך',
        'שלה',
        'שלנו',
        'שלכם',
        'שלכן',
        'שלהם',
        'שלהן',
        'לי',
        'לו',
        'לה',
        'לנו',
        'לכם',
        'לכן',
        'להם',
        'להן',
        'אותה',
        'אותו',
        'זה',
        'זאת',
        'אלה',
        'אלו',
        'תחת',
        'מתחת',
        'מעל',
        'בין',
        'עם',
        'עד',
        'נגר',
        'על',
        'אל',
        'מול',
        'של',
        'אצל',
        'כמו',
        'אחר',
        'אותו',
        'בלי',
        'לפני',
        'אחרי',
        'מאחורי',
        'עלי',
        'עליו',
        'עליה',
        'עליך',
        'עלינו',
        'עליכם',
        'לעיכן',
        'עליהם',
        'עליהן',
        'כל',
        'כולם',
        'כולן',
        'כך',
        'ככה',
        'כזה',
        'זה',
        'זות',
        'אותי',
        'אותה',
        'אותם',
        'אותך',
        'אותו',
        'אותן',
        'אותנו',
        'ואת',
        'את',
        'אתכם',
        'אתכן',
        'איתי',
        'איתו',
        'איתך',
        'איתה',
        'איתם',
        'איתן',
        'איתנו',
        'איתכם',
        'איתכן',
        'יהיה',
        'תהיה',
        'היתי',
        'היתה',
        'היה',
        'להיות',
        'עצמי',
        'עצמו',
        'עצמה',
        'עצמם',
        'עצמן',
        'עצמנו',
        'עצמהם',
        'עצמהן',
        'מי',
        'מה',
        'איפה',
        'היכן',
        'במקום שבו',
        'האם',
        'אם',
        'לאן',
        'למקום שבו',
        'מקום בו',
        'איזה',
        'מהיכן',
        'איך',
        'כיצד',
        'באיזו מידה',
        'מתי',
        'בשעה ש',
        'כאשר',
        'כש',
        'למרות',
        'לפני',
        'אחרי',
        'מאיזו סיבה',
        'הסיבה שבגללה',
        'למה',
        'מדוע',
        'לאיזו תכלית',
        'כי',
        'יש',
        'אין',
        'אך',
        'מנין',
        'מאין',
        'מאיפה',
        'יכל',
        'יכלה',
        'יכלו',
        'יכול',
        'יכולה',
        'יכולים',
        'יכולות',
        'יוכלו',
        'יוכל',
        'מסוגל',
        'לא',
        'רק',
        'אולי',
        'אין',
        'לאו',
        'אי',
        'כלל',
        'נגד',
        'אם',
        'עם',
        'אל',
        'אלה',
        'אלו',
        'אף',
        'על',
        'מעל',
        'מתחת',
        'מצד',
        'בשביל',
        'לבין',
        'באמצע',
        'בתוך',
        'דרך',
        'מבעד',
        'באמצעות',
        'למעלה',
        'למטה',
        'מחוץ',
        'מן',
        'לעבר',
        'מכאן',
        'כאן',
        'הנה',
        'הרי',
        'פה',
        'שם',
        'אך',
        'ברם',
        'שוב',
        'אבל',
        'מבלי',
        'בלי',
        'מלבד',
        'רק',
        'בגלל',
        'מכיוון',
        'עד',
        'אשר',
        'ואילו',
        'למרות',
        'אס',
        'כמו',
        'כפי',
        'אז',
        'אחרי',
        'כן',
        'לכן',
        'לפיכך',
        'מאד',
        'עז',
        'מעט',
        'מעטים',
        'במידה',
        'שוב',
        'יותר',
        'מדי',
        'גם',
        'כן',
        'נו',
        'אחר',
        'אחרת',
        'אחרים',
        'אחרות',
        'אשר',
        'או']


def get_functional_characters():
    return ['ב', 'ל', 'כ', 'ו', 'ה', 'ש', 'מ']
