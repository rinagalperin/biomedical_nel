# Hebrew Contextual Relevance Model For UMLS Entity Linking Using BERT

## Goal
Linking Hebrew terms to Unified Medical Language System (UMLs) entities can benefit text-analytics methods, make information retrieval and analysis 
easier and more accurate. This known problem is especially challenging in Hebrew since the language has different writing and sound systems compared to English. Our goal is to improve tagging results done by [MDTEL](https://github.com/yonatanbitton/mdtel)'s High Recall Matcher (HRM) using a Hebrew contextual relevance model.


## Our approach
We utlize **BERT** (Bidirectional Encoder Representations from Transformers), a technique for natural language processing (NLP) pre-training developed by Google.
Its key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modelling.
This is in contrast to previous efforts which looked at a text sequence either from left to right or combined left-to-right and right-to-left training.

## Hypothesis
Observation of the HRM output shows that in many cases it chooses candidate terms which are not actual UMLS entities, for example:

given the post:

>ברוכים הבאים לפורום דיכאון וחרדה  כאן תוכלו לשאול  להתייעץ וגם לייעץ בנושאי חרדה ודיכאון  עצב ותחושות נפשיות  מטרת הפורום בין היתר היא לסייע לחולים כרוניים באתר כמוני הנמצאים במצבי דיכאון וחרדה כתוצאה מהמצב הרפואי בו הם נתונים  ובעיקר שיהיה לבריאות מערכת כמוני


the HRM output contains the following match:
```
{"cand_match": "מוני", "umls_match": "מוני", "sim": 1.0, "cui": "C0590695", "match_eng": "Monit", "hebrew_key": "segmented_text", "match_tui": "T109", "semantic_type": "Chemical or drug", "all_match_occ": [162, 256], "curr_occurence_offset": 162}
```

the candidate match 'מוני' is mapped to UMLS 'Monit' which is a drug for chest pains. It was extracted from the word 'כ**מוני**' which is the name of the online forum.

Intuitively, using the context of the word in the post can give better indication of whether or not it is an actual medical term or not. In the above case, 'מערכת כמוני' can imply that 'כמוני' is a type/name of a system and not a drug.

We use context analysis, i.e.- combinations of one or more words that represent entities, phrases, concepts, and themes that appear in the text. The exact process we chose is described in the _Training data construction_ section.

## Training data construction
We use the output from HRM and the manual annotations, which can be found [here](https://drive.google.com/file/d/17JTxutH15P3R-Wd4x3d5ulY22KW0vVUC/view?usp=sharing) and attempt to use a contextual relevance language model to improve
the results of the tagged UMLS entities.

For each entry in the data (post from **Diabetes** community), we go over the matches found and do the following per-match:

1) <u>**word context**</u><br>
Using the offset of the term we find the original word from the text and create a window around it (depending on the chosen WINDOW_SIZE value) _to get the word context_ of the term (WINDOW_SIZE words to the left of the term and WINDOW_SIZE words to the right). 

2) <u>**UMLS from HRM**</u><br>
We collect the UMLS from the HRM output (under 'umls_match')

3) <u>**UMLS from manual annotations**</u><br>
We collect the term tagged in the mannual annotations that corresponds to the HRM match (using the offset) and look it up in the UMLS database to validate it's an actual UMLS (since in some cases it's not), so we take either the UMLS match or `Nan`.
<br><br>
_One-time runtime step: fix annotations data issues-_<br>
If there is no UMLS for the annotations' match, then we want to either insert the HRM UMLS instead or leave it as `Nan`. This is done during run time using a manual user-input method, where the user decides based on the context-window if the HRM's UMLS matches (`1` or `0` input). This process was done once and then we saved the fixed file separately (can be found [here](training_data/output_data/corrected_annotations.json)) and used it instead of the original manual annotations file. The process' [code](training_data/main.py) is commented out but can be repeated if necessary.<br><br>**For example**:<br> given the window:
>אם יש קשר בין הקרסול הנפוח לשאר אנא עזרתכם

&nbsp;&nbsp;&nbsp;```hrm_cui: C0174883, קרסול```<br>
&nbsp;&nbsp;&nbsp;```annotations' UMLS that has no corresponding CUI: קרסול הנפוח```<br>&nbsp;&nbsp;&nbsp;```is recall matcher right: 0```<br>
&nbsp;&nbsp;&nbsp;since in this case 'קרסול' is mapped to the drug Cresol (rather than the term 'ankle' or 'swollen ankle', as can be understood from the context).

4) <u>**Labels**</u><br>
For the given match we keep `1` as the label if the HRM UMLS matches with the annotations' UMLS, and `0` otherwise. Notice that if the annotations' UMLS is `Nan` at this point, it means that the HRM UMLS does not fit either and therefore will use label `0` as expected.

#### Utilizing BERT QA structure
The inputs come in the form of a **Context** / **Question** pair, and the outputs are **Answers**. We decided to utilize this structure to check if the HRM UMLS (**Question**) fits the context of the term (**Context**), where the manual annotations define the ground-truth label (**Answer** = labels from step 4 in the _Training data construction_ section).

#### Output
Final data output can be foud [here](training_data/output_data/training_data_4.json) (different outputs are created depending on the chosen window size).

The process results in 2821 examples which are split into train-test sets according to a 90%-10% division, respectively (2538 train examples, 283 test examples). 

## Results
The HRM's accuracy was **44.17%**.<br>
The following table summarizes our results on the test set:

(*) note that 'WINDOW_SIZE' represents the chosen number of words from each side (left and right) to the term.

| WINDOW_SIZE | Accuracy | Precision |  Recall | False negatives | False positives | True negatives | True positives |
|:-----------:|:--------:|:---------:|:-------:|:---------------:|:---------------:|:--------------:|:--------------:|
|      2      |  87.986% |  84.733%  |  88.8%  |        14       |        20       |       138      |       111      |
|      3      |  88.693% |  86.555%  | 86.555% |        16       |        16       |       148      |       103      |
|      4      |  85.866% |    87%    | 84.615% |        22       |        18       |       122      |       121      |


#### Example:<br>
based on the context of the sentece, 'חולים' is not a medical term but part of a named location: 'קופות החולים'. The HRM's mistake is caught by the contextual relevance model:

>המשלימים של קופות החולים המחיר גבוה יותר

```
HRM match: חולים
UMLS classifier answer: Wrong
Real answer: Wrong
```

## TODO
- Fine-tune model's hyper parameters.
- Train for 2 other communities: Depression and Sclerosis.

## Acknowledgements
+ Yonatan Bitton's [MDTEL](https://github.com/yonatanbitton/mdtel) work.
+ [Google's Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md).
+ Thanks to the [Ministry of Science and Technology](https://www.gov.il/he/departments/ministry_of_science_and_technology) for supporting our work
