import json

from training_data.utils import read_dict_from_json_file, clean_text

failed = [0]
total_annotated_cuis = [0]

# source: https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/


def create_mesh_and_id_mutual_mappings():
    mesh_data_file_path = "data/bc5cdr/desc2020.json"
    id_to_mesh_name, mesh_name_to_id = {}, {}
    for line in open(mesh_data_file_path, encoding="utf8", mode='r'):
        line_dict = json.loads(line)
        mesh_id = line_dict['id']
        name = clean_text(line_dict['name'])
        id_to_mesh_name[mesh_id] = name
        mesh_name_to_id[name] = mesh_id
    with open('data/bc5cdr/id_to_mesh_name_dict.json', mode='w', encoding='utf-8') as data_file:
        json.dump(id_to_mesh_name,
                  data_file,
                  ensure_ascii=False,
                  indent=4)
    with open('data/bc5cdr/mesh_name_to_id_dict.json', mode='w', encoding='utf-8') as data_file:
        json.dump(mesh_name_to_id,
                  data_file,
                  ensure_ascii=False,
                  indent=4)


def get_post_annotation(line, id_to_mesh_name_dict):
    if not('CID' in line):
        line_arr = line.split('\t')
        start_offset = line_arr[1]
        end_offset = line_arr[2]
        label = line_arr[4]
        cui = line_arr[5].split('\n')[0]
        umls = ''

        try:
            umls = id_to_mesh_name_dict[cui]
        except:
            failed[0] += 1
            pass

        total_annotated_cuis[0] += 1
        return {'term': umls,
                'start_offset': start_offset,
                'end_offset': end_offset,
                'label': label}  # example: label = "Chemical or drug"


def bc5cdr_reader(contents, id_to_mesh_name_dict):
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
        tokenized_document_text = clean_text(document_text)
        data['text'].append(document_text)
        data['tokenized_text'].append(tokenized_document_text)
        data['file_name'].append(curr_post_num)

        curr_line_num += 2

        # mentions + CUIs are located from the third line onwards (per document)
        # PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID \n
        curr_line = contents[curr_line_num].decode("utf-8")
        curr_line_pmid = str(curr_line).split('\t')[0]
        while curr_line_pmid == pmid:
            post_annotation = get_post_annotation(curr_line, id_to_mesh_name_dict)
            if post_annotation and post_annotation['term'] != '':
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


#create_mesh_and_id_mutual_mappings()
id_to_mesh_name_dict_file_path = 'data/bc5cdr/id_to_mesh_name_dict.json'
id_to_mesh_name_dict = read_dict_from_json_file(id_to_mesh_name_dict_file_path)

#bc5cdr_data_file = open("data/bc5cdr/CDR_TrainingSet.PubTator.txt", "rb")
bc5cdr_data_file = open("data/bc5cdr/CDR_TestSet.PubTator.txt", "rb")
contents = bc5cdr_data_file.readlines()

bc5cdr_data = bc5cdr_reader(contents, id_to_mesh_name_dict)

print(failed[0], 'annotated CUIs not found. Overall found: ', total_annotated_cuis[0]-failed[0])
with open('../training_data/json_files/annotations_data/bc5cdr_annotations_data_test', mode='w', encoding='utf-8') as data_file:
    json.dump(bc5cdr_data,
              data_file,
              ensure_ascii=False,
              indent=4)
