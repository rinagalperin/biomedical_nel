# Hebrew Contextual Relevance Model For UMLS Entity Linking Using BERT

## :dart: Goal
Linking Hebrew terms to Unified Medical Language System (UMLs) entities can benefit text-analytics methods, make information retrieval and analysis 
easier and more accurate. This known problem is especially challenging in Hebrew since the language has different writing and sound systems compared to English. Our goal is to improve tagging results done by [MDTEL](https://github.com/yonatanbitton/mdtel)'s High Recall Matcher (HRM) using a Hebrew contextual relevance model and utilizing the [manual annotations](https://drive.google.com/file/d/17JTxutH15P3R-Wd4x3d5ulY22KW0vVUC/view?usp=sharing).


## :game_die: Our approach
We utilize **BERT** (Bidirectional Encoder Representations from Transformers), a technique for natural language processing (NLP) pre-training developed by Google.
Its key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modelling.
This is in contrast to previous efforts which looked at a text sequence either from left to right or combined left-to-right and right-to-left training.

## :bulb: Hypothesis
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
{"cand_match": "טרשת נפוצה", "umls_match": "טרשת נפוצה", "sim": 1.0, "cui": "C0007795", "match_eng": "diffuse sclerosis", "hebrew_key": "post_txt", "match_tui": "T047", "semantic_type": "Disorder", "all_match_occ": [130], "curr_occurence_offset": 130}
```

 the candidate match 'טרשת נפוצה' is mapped to UMLS 'diffuse sclerosis' which is a disorder. 

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

Addressing these problems can result in higher precision and recall. This can be validated using the F1 metric.

### Solution
#### Problems 1, 2 solution
Intuitively, using the context of the word in the post can give better indication of whether or not it is an actual medical term or not, which addresses the first 2 problems:

1. 'מערכת כמוני' can imply that 'כמוני' is a type/name of a system and not a drug.

2. 'קהילת טרשת נפוצה' can imply that in the given context, 'טרשת נפוצה' is a type/name of an online community and not a disorder.

We use context analysis, i.e.- combinations of one or more words that represent entities, phrases, concepts, and themes that appear in the text. The exact process we implemented is described [here](#Data-construction).

Since such cases are not labeled as UMLS in the manual annotations, using the comparison between the HRM output and the annotations as the label provides the desired information to our BERT model during training (meaning that if the HRM _does tag such expressions_ - it is rightfully considered a mistake).

see the following guideline provided to the annotators as reference:

>Diabetes clinic<br>
מרפאת סכרת<br><br>
There isn't a disorder mention in this text. Diabetes is a disorder in the UMLS, but in this context it isn't meant as disease. Nobody suffers from the disease in this context. 

#### Problem 3 solution
 The 3rd problem however, is an inherent limitation of the HRM as it can't choose terms which have no corresponding CUI in the UMLS database and therefore a modification of the HRM logic itself is required:

An important observation is that in the Hebrew language, adding more information to nouns (making them more specific) is done by adding more words following the general term using preposition or construct state, for example:
>'חיסון לשפעת' (preposition)

 >'קליניקה לטיפול בסכרת' (preposition)

 >'חולה סרטן הדם' (construct state)

 This means that considering the given term tagged by the HRM combined with one or two of the words following it - can provide the necessary medical terms that lack a CUI. 


## :construction: Training
### Input
We use the output from HRM and the manual annotations, which can be found [here](https://drive.google.com/file/d/17JTxutH15P3R-Wd4x3d5ulY22KW0vVUC/view?usp=sharing) and attempt to use a contextual relevance language model to improve
the results of the tagged UMLS entities. We essentially implement a patch for the HRM, adding more tagged ter

Intial dataset description:
- 258 posts
- vocabulary of size 7,094
- 734 unique terms tagged by HRM
- overall terms frequencies:

| Term Length (words count) | Frequency |
|:-------------------------:|:---------:|
|             1             |    2331   |
|             2             |    397    |
|             3             |     90    |
 
### Data construction
For each entry in the data (post from **Diabetes** community), we go over the matches found and do the following per-match:

1) <u>**word context**</u><br>
Using the offset of the term we find the original word from the text and create a window around it (depending on the chosen WINDOW_SIZE value) _to get the word context_ of the term (WINDOW_SIZE words to the left of the term and WINDOW_SIZE words to the right). 

2) <u>**UMLS from HRM**</u><br>
We collect the UMLS from the HRM output (under 'umls_match')

3) <u>**UMLS from manual annotations**</u><br>
We collect the term tagged in the manual annotations that corresponds to the HRM match (using the offset)  or `Nan` if there isn't one.

4) <u>**Labels**</u><br>
For the given match we keep `1` as the label if either the HRM UMLS (including the non-CUI terms that we synthetically added as described [here](#Problem-3-solution)) matches with the annotations' UMLS or if the HRM candidate match (under 'cand_match') matches with the annotations' UMLS, and `0` otherwise. 
  
 Comparing the HRM candidate match with the annotations' UMLS mitigates small irrelevant deviations: since the annotations' UMLS use the exact syntax from the text, if the HRM found a CUI match with the candidate that is identical to the annotations' UMLS - then that CUI must in turn fit the annotations' UMLS as well. 
In this comparison we allow a difference in at most the first character for any one of each expression's words, considering term-pairs to be identical in cases such as the following:
> האינסולין, אינסולין

 > לטמוקסיפן, טמוקסיפן

 > בטבליות, טבליות

 > לרמות הסוכר, רמות סוכר

 this step filtered **over 50%** of mismatches!

### Output
Final data output can be found [here](training_data/output_data/training_data_4.json) (different outputs are created depending on the chosen window size).

## :microscope: Testing
### Data
Instead of splitting the output of the [data construction process](#Data-construction) into training/testing sets, we split the [input](#input) (i.e. - the HRM output) into such sets (90% training, 10% testing) and then perform the data construction on each set. This helps avoid overfitting.

About 12% of unique terms in the test set have been seen during training, however, since they appear in novel contexts it is important to test the model's answer to them.

### Utilizing BERT QA structure
The inputs come in the form of a **Context** / **Question** pair, and the outputs are **Answers**. We decided to utilize this structure to check if the HRM UMLS (**Question**) fits the context of the term (**Context**), where the manual annotations define the ground-truth label (**Answer** = labels from step 4 in the [data construction process](#Data-construction) section).

## :bar_chart: Results
The HRM's accuracy was **44.17%**.<br>
MDTEL's F1-measure on Diabetes community was **73%**.
<br><br>
The following tables summarize our results for different versions of our model. Each version tested against both the baseline test set and the expanded one (containing new terms):

(*) note that 'WINDOW_SIZE' represents the chosen number of words from each side (left and right) to the term.

### Baseline BERT (no HRM expansion)
#### baseline test set
| WINDOW_SIZE | Accuracy | Precision | Recall | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:--------:|:---------:|:------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |    85%   |    91%    |   80%  |        32       |        13       |       129      |       133      |     85.15%   |
|      3      |    84%   |    82%    |   85%  |        20       |        23       |       118      |       110      |     83.47%   |
|      4      |   87.7%  |  86.667%  |   89%  |        11       |        14       |       87       |       91       |     87.8%    |


#### expanded test set
| WINDOW_SIZE | Accuracy | Precision | Recall | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:--------:|:---------:|:------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |    85%   |    91%    |   80%  |        32       |        13       |       129      |       133      |     85.15%   |
|      3      |    84%   |    82%    |   85%  |        20       |        23       |       118      |       110      |     83.47%   |
|      4      |   87.7%  |  86.667%  |   89%  |        11       |        14       |       87       |       91       |     87.8%    |


#### Inference example:<br>
based on the context of the sentence, 'חולים' is not a medical term but part of a named location: 'קופות החולים'. The HRM's mistake is caught by the contextual relevance model:

>המשלימים של קופות החולים המחיר גבוה יותר

```
HRM match: חולים
UMLS classifier answer: Wrong
Real answer: Wrong
```

### BERT model w/ HRM expansion

#### baseline test set

| WINDOW_SIZE | Accuracy | Precision | Recall | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:--------:|:---------:|:------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |   93.5%  |   86.8%   |   79%  |        21       |        12       |       395      |       79       |     82.72%   |
|      3      |   88.4%  |    74%    |   60%  |        86       |        45       |       871      |       129      |     66.3%    |
|      4      |   91.6%  |   80.2%   |   77%  |        30       |        25       |       501      |       101      |     78.56%   |

#### expanded test set

| WINDOW_SIZE | Accuracy | Precision | Recall | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:--------:|:---------:|:------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |   93.5%  |   86.8%   |   79%  |        21       |        12       |       395      |       79       |     82.72%   |
|      3      |   88.4%  |    74%    |   60%  |        86       |        45       |       871      |       129      |     66.3%    |
|      4      |   91.6%  |   80.2%   |   77%  |        30       |        25       |       501      |       101      |     78.56%   |

#### Inference example:<br>
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

- [x] modify HRM to accept medical terms which have no corresponding CUI
- [ ] Fine-tune model's hyper parameters.
- [ ] Train for 2 other communities: Depression and Sclerosis.


## :high_brightness: Acknowledgements
+ Yonatan Bitton's [MDTEL](https://github.com/yonatanbitton/mdtel) work.
+ [Google's Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md).
+ Thanks to the [Ministry of Science and Technology](https://www.gov.il/he/departments/ministry_of_science_and_technology) for supporting our work

## :email: Contact
- rinag@post.bgu.ac.il
- schnapp@post.bgu.ac.il
