# Cross-Lingual UMLS Named Entity Linking Using UMLS Dictionary Fine-Tuning

## :dart: Goal
Linking terms to Unified Medical Language System (UMLS) entities can benefit text-analytics methods and make information retrieval and analysis 
easier and more accurate. UMLS concepts are mostly labeled in English, therefore NEL is especially challenging when the source language is different (e.g., Hebrew). We aim for a general solution that can be adapted to any source language. We operate in a low resource setting, where the ontology is large, text describing most entities is not available, and labeled data can only cover a small portion of the ontology. We also consider different genres of text to be annotated, ranging from consumer health medical articles in popular web sites to scientific biomedical articles.

## :gift: Our contribution
(+) We provide a general framework for cross-lingual UMLS NEL that can be adapted to source languages with few pre-requisites; our method includes four steps (a) offline unsupervised learning of a language-specific UMLS dictionary; for each document: (b) generation of candidate mentions, (c) high-recall matching of candidate mentions to UMLS concepts and (d) contextual relevance filtering of (candidate, concept) pairs.  Steps (c) and (d) take advantage of multi-lingual pre-trained transformer language models (PLMs). (+) Our method exploits a small annotated corpus of documents in the source language and genre annotated manually for UMLS mentions (a few thousands annotated mentions).  This training data is split to support (a) the extension of the unsupervised UMLS dictionary with corpus-salient entity names and (b) fine-tune the contextual ranking and filtering of (candidate mentions, concept) pairs.  We find that the step of \techniquename~ boosts NEL performance and identify a clear tradeoff in allocating training data between lexicon extension and contextual fine-tuning.
(+) We demonstrate results of our approach on both Hebrew and English.
We achieve new SOTA on the Hebrew Camoni corpus [bitton2020](https://academic.oup.com/jamia/article/27/10/1585/5903800?login=true) with +8.87 F1 
and on the English dataset MedMentions [mohan2019medmentions](https://arxiv.org/abs/2101.10587) with +7.3 F1

## :bulb: Approach
Our end-to-end xNEL model consists of four consecutive stages: (1)  **multilingual UMLS mapping**: generate UMLS dictionary C_L based on the method of [bitton2020](https://academic.oup.com/jamia/article/27/10/1585/5903800?login=true); (2) **candidate generation**: consider all spans of up to k words as candidate mentions and compute vector representations for both mentions and concepts; (3) **high recall matching**: use a semantic similarity based score function to generate the top matching entities with high recall and (4) **contextual relevance modeling**: encode each candidate into a context-dependent vector representation using a pre-trained transformer-based language model fine tuning process. 

![Alt text](pipeline.png?raw=true "Full pipeline overview")

## :bar_chart: Results
- Camoni Corpus

|       Model                                           |   Community |  Accuracy%  | Precision% |   Recall%       |  F1 measure%    |
|:-----------------------------------------------------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|
|  [MDTEL](https://github.com/yonatanbitton/mdtel)      |  Diabetes   |  **97.0**   |    71.0    |  **75.0**       |    73.0         |
|*Our model*                                            |  Diabetes   |    89.2     |  **98.3**  |    73.8         |    **84.3**     |


|       Model |   Community |  Accuracy%  | Precision% |   Recall%       |  F1 measure%    |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|
|  MDTEL      | Depression  |  **99.0**   |   77.0     |    73.0         |    75.0         |
|*Our model*  | Depression  |   90.8      |   97.7     |  **76.9**       |    **86.0**     |


|       Model |   Community |  Accuracy%  | Precision% |   Recall%       |  F1 measure%    |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|
|  MDTEL      | Sclerosis   |  **98.0**   |   82.0     |  **71.0**       |    76.0         |
|*Our model*  | Sclerosis   |   86.3      | **98.3**   |    67.8         |    **80.3**     |

- MedMentions

|           Model                                      |   Accuracy  |  Precision  |   Recall   |   F-score   |
|:----------------------------------------------------:|:-----------:|:-----------:|:----------:|:-----------:|
|      TaggerOne                                       |             |     47.1    |    43.6    |     45.3    | 
|      MedLinker                                       |             |     48.4    |    50.1    |     49.2    | 
|[LRR](https://arxiv.org/abs/2101.10587)               |             |     63.0    |    52.0    |     57.0    | 
|      *Our model*                                     |     74.8    |  **76.4**   |  **55.5**  |   **64.3**  |

- BC5CDR

|    Model      |   Dataset    |   F-score   |
|:-------------:|:------------:|:-----------:|
|    BioBERT    |     BC5CDR   |     88.6    |
|    SciBERT    |     BC5CDR   |     90.0    |
|    SapBERT    |     BC5CDR-d |  **93.5**   |
|  *Our model*  |     BC5CDR   |     73.0    |

## :high_brightness: Acknowledgements
+ This work was funded by the Ministry of Science and Technology scholarship for STEM research students.

## :email: Contact
- rinagalperin@gmail.com
