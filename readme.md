# Hebrew Contextual Relevance Model For UMLS Entity Linking Using BERT

## :dart: Goal
Linking Hebrew terms to Unified Medical Language System (UMLs) entities can benefit text-analytics methods and make information retrieval and analysis 
easier and more accurate. This known problem is especially challenging in Hebrew since the language has different writing and sound systems compared to English. Our goal is to improve tagging results done by [MDTEL](https://github.com/yonatanbitton/mdtel)'s High Recall Matcher (HRM) using a Hebrew contextual relevance model and utilizing the [manual annotations](https://drive.google.com/file/d/17JTxutH15P3R-Wd4x3d5ulY22KW0vVUC/view?usp=sharing) as ground truth.


## :gift: Our contribution
We utilize **BERT** (Bidirectional Encoder Representations from Transformers), a technique for natural language processing (NLP) pre-training developed by Google.
Its key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modelling. To do so, we implement a reduction from the HRM output to a different data structure that can be used as an input to BERT. Moreover, we perform data augmentation to improve our learning model's performance.

## :bulb: Hypothesis & approach
### Problem identification
Observation of the HRM output reveals 3 main problems:

#### Problem 1
The HRM chooses candidate terms which are not actual UMLS entities, for example:

 given the post:
>ברוכים הבאים לפורום דיכאון וחרדה  כאן תוכלו לשאול  להתייעץ וגם לייעץ בנושאי חרדה ודיכאון  עצב ותחושות נפשיות  מטרת הפורום בין היתר היא לסייע לחולים כרוניים באתר כמוני הנמצאים במצבי דיכאון וחרדה כתוצאה מהמצב הרפואי בו הם נתונים  ובעיקר שיהיה לבריאות **מערכת כמוני**

 the HRM output contains the following match:
```
{"cand_match": "מוני", "umls_match": "מוני", "sim": 1.0, "cui": "C0590695", "match_eng": "Monit", "hebrew_key": "segmented_text", "match_tui": "T109", "semantic_type": "Chemical or drug", "all_match_occ": [162, 256], "curr_occurence_offset": 162}
```

 the candidate match 'מוני' is mapped to UMLS 'Monit' which is a drug for chest pains. It was extracted from the word 'כ**מוני**' which is the name of the online forum. 

#### Problem 2
The HRM chooses candidate terms which are not medical terms when considering the context, for example:

 given the post:
 >בעקבות הגעתן הקרובה של התרופות האוראליות החדשות  העלנו סקר אשר יציג את עמדתכן בנושא  את הסקר ניתן לראות בתחתית עמוד הבית של **קהילת טרשת נפוצה**

 The HRM output contains the following match:
```
{"cand_match": "טרשת נפוצה", "umls_match": "טרשת נפוצה", "sim": 1.0, "cui": "C0007795", "match_eng": "multiple sclerosis", "hebrew_key": "post_txt", "match_tui": "T047", "semantic_type": "Disorder", "all_match_occ": [130], "curr_occurence_offset": 130}
```

 the candidate match 'טרשת נפוצה' is mapped to UMLS 'multiple sclerosis' which is a disorder. 

#### Problem 3
The HRM chooses candidate terms which are not the full medical terms due to its limitation of only being able to choose terms that have a corresponding CUI, i.e. - are part of the UMLS database. For example:

 given the post:
 >שלום  אני חולה סוכרת ורציתי לדעת מהם היתרונות של משאבת אינסולין אומניפוד ומהם חסרונותיה  כמו כן רציתי לדעת האם יש סיכון ב**חיסון שפעת חזירים** לחולי סוכרת תודה

 The HRM output contains the following match:
```
{"cand_match": "חיסון", "umls_match": "חיסון", "sim": 1.0, "cui": "C0042210", "match_eng": "Vaccines", "hebrew_key": "segmented_text", "match_tui": "T116", "semantic_type": "Chemical or drug", "all_match_occ": [121], "curr_occurence_offset": 121}
```

 the candidate match 'חיסון' is mapped to the general UMLS 'Vaccines' which is a chemical or drug, rather than selecting the more specific medical term 'חיסון שפעת חזירים'.

 Other examples include:
 >אני **חולה לב** סכרת וכלוסטרול גבוה ונוטל מספר כדורים ברצוני לדעת על השילוב בניהם

 where the closest UMLS with documented CUI is 'חולה'.

 >אנסה להסביר בצורה פשוטה ה**חומצה אלפא לינולנית** עוברת בגוף היונקים

 where the closest UMLS with documented CUI is 'חומצה'.

 Since the annotators are medical professionals with years of theoretical and practical knowledge, we treat their annotations as the ground-truth, even if the terms have no corresponding CUI in the UMLS database. Moreover, the medical field is constantly changing and therefore we would like our model to have good generalization ability in order to be able to identify new terms, such as '2019 coronavirus vaccination', which didn't exist when the manual annotations were made.

Overall we went over about 200 posts from the three communities (diabetes, sclerosis, depression) and their corresponding HRM tags and found that these problems occur quite often (about 20% of the posts were related to one or more of the above tagging mistakes). 

Addressing these problems can result in higher precision and recall.

### Solution
#### Problems 1, 2 solution
Intuitively, using the context of the word in the post can give better indication of whether it is an actual medical term or not, which addresses the first 2 problems:

1. 'מערכת כמוני' can imply that 'כמוני' is a type/name of a system and not a drug.

2. 'קהילת טרשת נפוצה' can imply that in the given context, 'טרשת נפוצה' is a type/name of an online community and not a disorder.

We use context analysis, i.e.- combinations of one or more words that represent entities, phrases, concepts, and themes that appear in the text. The exact process we implemented is described [here](#Data-construction).

Since such cases are not labeled as UMLS in the manual annotations, using the comparison between the HRM output and the annotations as the label provides the desired information to our BERT model during training (meaning that if the HRM _does tag such expressions_ - it is rightfully considered a mistake).

see the following guideline provided to the annotators as reference:

>Diabetes clinic<br>
מרפאת סכרת<br><br>
There isn't a disorder mention in this text. Diabetes is a disorder in the UMLS, but in this context it isn't meant as disease. Nobody suffers from the disease in this context. 

#### Problem 3 solution
 The 3rd problem however, is an inherent limitation of the HRM as it can't choose terms which have no corresponding CUI in the UMLS database and therefore a modification of the HRM output is required.

An important observation is that in the Hebrew language, adding more information to nouns (making them more specific) is done by adding more words following the general term using preposition, construct state or a relative clause, for example:
>'טבליות להורדת הסוכר' (preposition)

 >'חולה סרטן הדם' (construct state)

 This means that considering the term tagged by the HRM (e.g. - 'חולה', 'טבליות') combined with a few of the words following it - can provide the necessary medical terms that lack a CUI. We essentially implement a patch for the HRM, adding more tagged terms.


## :construction: Training
### Input
We use the output from the HRM and the manual annotations regarding the **Diabetes** community (which can be found [here](https://drive.google.com/file/d/17JTxutH15P3R-Wd4x3d5ulY22KW0vVUC/view?usp=sharing)) to construct the training and testing data for our contextual relevance model. 

Intial dataset description:
- 258 posts
- 22,677 tokens
- vocabulary of size 7,094 (unique tokens)
- 734 unique terms tagged by HRM
- tagged terms frequencies:

| Term Length (words count) | Frequency |
|:-------------------------:|:---------:|
|             1             |    2331   |
|             2             |    397    |
|             3             |     90    |
 
### Data construction
For each entry in the data (post), we go over the matches found and do the following per-match:

1) <u>**word context**</u><br>
Using the offset of the term we find the original word from the text and create a window around it (depending on the chosen WINDOW_SIZE value) _to get the word context_ of the term (WINDOW_SIZE words to the left of the term and WINDOW_SIZE words to the right). 

2) <u>**UMLS from HRM**</u><br>
We collect the UMLS from the HRM output (under 'umls_match').

3) <u>**UMLS from manual annotations**</u><br>
We collect the term tagged in the manual annotations that corresponds to the HRM match (using the offset)  or `Nan` if there isn't one.

4) <u>**Labels**</u><br>
For the given match we keep `1` as the label if either the HRM UMLS matches the annotations' UMLS or if the HRM candidate match (under 'cand_match') matches the annotations' UMLS, and `0` otherwise.<br><br>Comparing the HRM candidate match with the annotations' UMLS mitigates small irrelevant deviations: since the annotations' UMLS use the exact syntax from the text, if the HRM found a CUI match with the candidate that is identical to the annotations' UMLS - then that CUI must in turn fit the annotations' UMLS as well. 
In this comparison we allow a difference in at most the first character (provided that it is a functional character) for any one of each expression's words, considering term-pairs to be identical in cases such as the following:

   > האינסולין, אינסולין

   > לטמוקסיפן, טמוקסיפן

   > בטבליות, טבליות

   > לרמות הסוכר, רמות סוכר

   this step filtered **over 50%** of mismatches!
   
   Step 4 results in <b>"baseline train/test sets"</b>, containing about 1,400/150 examples, respectively (changes each run)
 
 5) <u>**Data augmentation**</u><br>
 For the non-baseline model we expand the UMLS from the HRM to consider non-CUI terms and synthetically add them as examples for our BERT model (as described [here](#Problem-3-solution)). For each expanded term, we repeat steps 1 and 4.
 <br>Step 5 results in <b>"expanded train/test sets"</b>, containing about 4,000/800 examples, respectively (changes each run)

### Output
Final data output can be found [here](training_data/output_data/training_data_4.json) (different outputs are created depending on the chosen window size).

### BERT model
BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.
Pre-training BERT is fairly expensive but it is a one-time procedure for each language so we can utilize Google's multilingual BERT and avoid training our own model from scratch.

 The multilingual BERT model was pre-trained on the top 100 languages with the largest Wikipedias. The entire Wikipedia dump for each language (excluding user and talk pages) was taken as the training data for each language.

BERT can be adapted to many types of NLP tasks very easily with an inexpensive fine-tuning process that provides state-of-the-art results on sentence-level (e.g., SST-2), sentence-pair-level (e.g., MultiNLI), word-level (e.g., NER), and span-level (e.g., SQuAD) tasks with almost no task-specific modifications.

 We chose to utilize BERT's QA structure for fine tuning where the inputs come in the form of a **Context** / **Question** pair, and the outputs are **Answers**. We decided to use this structure to check if the HRM UMLS (i.e. - **Question**) fits the context of the term (i.e. - **Context**), where the manual annotations define the ground-truth label (i.e. - **Answer** = labels retrieved from step 4 in the [data construction process](#Data-construction) section).

## :microscope: Testing and evaluation
### Data
Instead of splitting the output of the [data construction process](#Data-construction) into training/testing sets, we split the [input](#input) (i.e. - the HRM output) into such sets (90% training, 10% testing) and then perform the data construction on each set. This helps avoid overfitting.

About 12% of unique terms in the test set have been seen during training, however, since they appear in different contexts - it is important to test the model's answer to them.

## :bar_chart: Intrinsic evaluation results (UMLS tagging)
### HRM results (baseline)

- 22,677 total tokens (TP+FP+FN+TN)
- 1,653 total manual annotations
- 2,818 total tagged by HRM

- 1,262 correctly tagged by HRM (TP)
- 2,818-1,262 = 1,556 (FP)
- 1,653-1,262 = 391 (FN, the terms that the HRM didn't tag)

- Precision=TP/TP+FP=1,262/(1,262+1,556)
- Recall=TP/TP+FN=1,262/(1,262+391)

|  Precision  |     Recall    |  F1 measure  |
|:-----------:|:-------------:|:------------:|
|    44.17%   |    76.34%     |    55.96%    |

### HRM results (expanded data)
Since we perform data augmentation and add at least one more FP match for each HRM match, we expect to get lower precision but enhanced recall, with our model's goal being to fix the precision.

- 8,454 total tagged by HRM + expansion

- 1,536 correctly tagged by HRM + expansion (TP)
- 8,454-1,536 = 6,918(FP)
- 1,653-1,536 = 117 (FN)

- Precision=TP/TP+FP=1,536/(1,536+6,918)
- Recall=TP/TP+FN=1,262/(1,262+391)

|  Precision  |     Recall    |  F1 measure  |
|:-----------:|:-------------:|:------------:|
|    18.17%   |     92.9%     |     30.4%    |

### MDTEL's results
|  Precision  |     Recall    |  F1 measure  |
|:-----------:|:-------------:|:------------:|
|      71%    |      75%      |      73%     |

### Our results
The following tables summarize our results for different versions of our model, comparing our performance to the HRM.
Each version tested against both the baseline test set and the expanded one (containing new terms):

(*) note that 'WINDOW_SIZE' represents the chosen number of words from each side (left *and* right) to the term.

#### Baseline BERT (no HRM expansion in [step 5](#Data-construction))
##### baseline test set
| WINDOW_SIZE |   Accuracy  |  Precision  |   Recall   | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |     89%     |    85.3%    |    84.7%   |        22       |        21       |       227      |       122      |      85%     |
|      3      |    88.8%    |    84.2%    |    85.4%   |        21       |        23       |       225      |       123      |     84.8%    | 
|      4      |    89.3%    |    85.4%    |    85.4%   |        21       |        21       |       227      |       123      |     85.4%    |


##### expanded test set
| WINDOW_SIZE |   Accuracy  |  Precision  |   Recall   | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |    36.5%    |    18.5%    |    88.2%   |       22        |        725      |       264      |       165      |     30.6%    |
|      3      |    36.5%    |    18.5%    |    88.2%   |       22        |        725      |       264      |       165      |     30.6%    |
|      4      |    40.5%    |    19.6%    |    88.2%   |       22        |        677      |       312      |       165      |     32.1%    |


##### Inference example:<br>
based on the context of the sentence, 'חולים' is not a medical term but part of a named location: 'קופות החולים'. The HRM's mistake is caught by the contextual relevance model:

>המשלימים של קופות החולים המחיר גבוה יותר

```
HRM match: חולים
UMLS classifier answer: Wrong
Real answer: Wrong
```

#### BERT model w/ HRM expansion

##### baseline test set

| WINDOW_SIZE |   Accuracy  |  Precision  |   Recall   | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |     89%     |    82.2%    |    89.6%   |        15       |        28       |       220      |       129      |     85.7%    |
|      3      |     87%     |     80%     |     86%    |        20       |        31       |       217      |       124      |     82.9%    |
|      4      |     87%     |     80%     |     87%    |        19       |        31       |       217      |       125      |     83.3%    |

##### expanded test set

| WINDOW_SIZE |   Accuracy  |  Precision  |   Recall   | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |     91.2%   |     67.5%   |    85.6%   |        27       |        77       |       912      |       160      |     75.5%    |
|      3      |     89.7%   |     62.8%   |    86.6%   |        25       |        96       |       893      |       162      |     72.8%    |
|      4      |     89.5%   |     62.3%   |    85.6%   |        27       |        97       |       892      |       160      |     72.1%    |

##### Inference example:<br>
'סוכר בדם' is a medical term tagged by the manual annotators but originally not tagged by the HRM. After synthetically adding more matches to the HRM, this term is now correctly identified by our model:

>את רמת הסוכר בדם מסקנה לנסות

```
HRM match: הסוכר בדם
UMLS classifier answer: Right
Real answer: Right
```

we also tested the model's answer to a novel sentence to see if it can generelize well so that new medical terms are not ignored (as described in [the overview of the HRM's limitations](#Problem-3)):
>האם קיים חיסון לקורנה שיכול לעזור לי

```
HRM match: חיסון לקורונה
UMLS classifier answer: Right
```

when testing the same sentence but with a less precise HRM match ('חיסון' as oppose to 'חיסון לקורונה'), the model rightfully tags it as wrong:
>האם קיים חיסון לקורנה שיכול לעזור לי

```
HRM match: חיסון
UMLS classifier answer: Wrong
```

## :hammer: TODO 

- [x] Augment HRM output with medical terms which have no corresponding CUI
- [ ] Train for 2 other communities: Depression and Sclerosis.


## :high_brightness: Acknowledgements
+ Yonatan Bitton's [MDTEL](https://github.com/yonatanbitton/mdtel) work.
+ [Google's Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md).
+ Thanks to the [Ministry of Science and Technology](https://www.gov.il/he/departments/ministry_of_science_and_technology) for supporting our work

## :email: Contact
- rinag@post.bgu.ac.il
- schnapp@post.bgu.ac.il
