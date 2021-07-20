# PubMentions format (source: https://github.com/chanzuckerberg/MedMentions)
# PMID | t | Title text
# PMID | a | Abstract text
# PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID

# example:
# 25763772|t|DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis
# 25763772|a|Pseudomonas aeruginosa (Pa) infection in cystic fibrosis (CF) patients is associated with worse long-term pulmonary disease and shorter survival, and chronic Pa infection (CPA) is associated with reduced lung function, faster rate of lung decline, increased rates of exacerbations and shorter survival. By using exome sequencing and extreme phenotype design, it was recently shown that isoforms of dynactin 4 (DCTN4) may influence Pa infection in CF, leading to worse respiratory disease. The purpose of this study was to investigate the role of DCTN4 missense variants on Pa infection incidence, age at first Pa infection and chronic Pa infection incidence in a cohort of adult CF patients from a single centre. Polymerase chain reaction and direct sequencing were used to screen DNA samples for DCTN4 variants. A total of 121 adult CF patients from the Cochin Hospital CF centre have been included, all of them carrying two CFTR defects: 103 developed at least 1 pulmonary infection with Pa, and 68 patients of them had CPA. DCTN4 variants were identified in 24% (29/121) CF patients with Pa infection and in only 17% (3/18) CF patients with no Pa infection. Of the patients with CPA, 29% (20/68) had DCTN4 missense variants vs 23% (8/35) in patients without CPA. Interestingly, p.Tyr263Cys tend to be more frequently observed in CF patients with CPA than in patients without CPA (4/68 vs 0/35), and DCTN4 missense variants tend to be more frequent in male CF patients with CPA bearing two class II mutations than in male CF patients without CPA bearing two class II mutations (P = 0.06). Our observations reinforce that DCTN4 missense variants, especially p.Tyr263Cys, may be involved in the pathogenesis of CPA in male CF.
# 25763772        0       5       DCTN4   T116,T123    C4308010

# MentionTextSegment: the actual mention between StartIndex and EndIndex (mention)
# SemanticTypeID: id for the Semantic Type that entity is linked to in UMLS (T047, T116, T123, etc.)
# EntityID: the CUI, UMLS entity (concept) id

# corpus statistics: https://github.com/chanzuckerberg/MedMentions/tree/master/full
import json
import gzip

from training_data.utils import read_dict_from_json_file, clean_text

failed = [0]
total_annotated_cuis = [0]


def get_post_annotation(line, cui_to_umls_dict):
    # line example: ['25763772', '0', '5', 'DCTN4', 'T116,T123', 'C4308010\n']
    line_arr = line.split('\t')
    start_offset = line_arr[1]
    end_offset = line_arr[2]
    cui = line_arr[5].split('\n')[0]
    umls = ''

    try:
        umls = cui_to_umls_dict[cui][0]
    except:
        failed[0] += 1
        pass

    total_annotated_cuis[0] += 1

    return {'term': umls,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'label': 0}  # example: label = "Chemical or drug"


def med_mentions_reader(contents, cui_to_umls_dict):
    data = {'text': [], 'tokenized_text': [], 'file_name': [], 'merged_inner_and_outer': []}

    curr_post_num = 0
    curr_line_num = 0

    # going over the documents
    while curr_line_num < len(contents):
        curr_line = contents[curr_line_num].decode("utf-8")
        annotations = []  # new document annotations
        pmid = str(curr_line).split('|')[0]
        title = curr_line.split('|')[-1]
        abstract = contents[curr_line_num + 1].decode("utf-8").split('|')[-1]
        document_text = title + ' ' + abstract  # concatenating the Title and Abstract, separated by a SPACE character

        data['text'].append(document_text)
        data['tokenized_text'].append(clean_text(document_text))
        data['file_name'].append(curr_post_num)

        curr_line_num += 2

        # mentions + CUIs are located from the third line onwards (per document)
        curr_line = contents[curr_line_num].decode("utf-8")
        curr_line_pmid = str(curr_line).split('\t')[0]
        while curr_line_pmid == pmid:
            post_annotation = get_post_annotation(curr_line, cui_to_umls_dict)
            if post_annotation['term'] != '':
                annotations.append(post_annotation)
            curr_line_num += 1
            next_line_raw = contents[curr_line_num]
            if next_line_raw and next_line_raw.decode("utf-8") != '\n':
                next_line = next_line_raw.decode("utf-8")
                next_line_pmid = str(next_line).split('\t')[0]
                curr_line_pmid = next_line_pmid
                curr_line = next_line
            else:
                curr_line_pmid = -1

        data['merged_inner_and_outer'].append(annotations)
        curr_line_num, curr_post_num = curr_line_num + 1, curr_post_num + 1

    return data


medmentions_data_file = gzip.open("data/medmentions/corpus_pubtator.txt.gz", "rb")
contents = medmentions_data_file.readlines()

#cui_to_umls_dict_file_path = 'data/cui_to_umls.json'
#file_name = '../training_data/json_files/annotations_data/med_mentions_annotations_data'

cui_to_umls_dict_file_path = 'C:/Users/Rins/Downloads/UMLSParser-master/UMLSParser-master/umlsparser/cui_to_umls.json'
file_name = 'E:/nlp_model/hrm/medmentions/med_mentions_annotations_data'

cui_to_umls_dict_from_file = read_dict_from_json_file(cui_to_umls_dict_file_path)
med_mentions_data = med_mentions_reader(contents, cui_to_umls_dict_from_file)

print(failed[0], 'annotated CUIs not found. Overall found: ', total_annotated_cuis[0]-failed[0])
with open(file_name, mode='w', encoding='utf-8') as data_file:
    json.dump(med_mentions_data,
              data_file,
              ensure_ascii=False,
              indent=4)
