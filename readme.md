# Contextual Relevance Model For Cross-Lingual UMLS Entity Linking Using UMLS Dictionary Fine-Tuning

## :dart: Goal
Linking terms to Unified Medical Language System (UMLS) entities can benefit text-analytics methods and make information retrieval and analysis 
easier and more accurate. UMLS concepts are mostly labeled in English, therefore NEL is especially challenging when the source language is different (e.g., Hebrew). This requires a general solution that can be easily adapted to any source language.  We address contextual relevance as an important factor when performing NEL and isolate the language-dependant components required for the task.  We operate in a low resource  setting (zero-shot),  where the ontologies are large, there is a lack of descriptive text  defining most entities, and labeled data can only cover a small portion of the ontology. This means we do not rely on any entity description resources. 

## :gift: Our contribution
+ We provide a general framework for cross-lingual UMLS NEL that can easily be adapted to any source language. While we focus here on UMLS NEL, our solution can be applied to solve other NEL tasks.
+ We introduce a novel enrichment technique that can help boost NEL performance.
+ We demonstrate results of our approach on both Hebrew and English as the source languages and compare to state-of-the-art (SOTA) solutions which are language-specific rather than multilingual. We improve on SOTA results on the Hebrew Camoni corpus, achieving +8.87 F1 on average across the three communities in the dataset. We also achieve SOTA on English dataset MedMentions, +4.4 F1 over the latest best result.

## :bulb: Approach
Our proposed end-to-end NEL model consists of four consecutive stages: (1)  **multilingual UMLS mapping**; generate a list <em>U_L</em> of medical concepts in language <em>L</em> mapped to universal CUIs from <em>C</em>, (2) **candidate generation**; consider all spans as candidate mentions and compute vector representations for both mentions and concepts, (3) **high recall matching**; use a semantic similarity based score function to generate the top matching entities with high recall being the priority and (4) **contextual relevance modeling**; embed each candidate span in its context, defined by the tokens surrounding it in the original text, alongside the label of whether or not the given mention is a medical concept in this context. This is used as an input to the BERT model fine-tuning.

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
|      TaggerOne                                       |             |     47.1    |    43.6    |     45.4    | 
|      MedLinker                                       |             |     48.4    |    50.1    |     49.2    | 
|[LRR](https://arxiv.org/abs/2101.10587)               |             |     63.0    |    52.0    |     57.0    | 
|      *Our model*                                     |     73.8    |  **76.4**   |  **55.5**  |   **64.3**  |

- BC5CDR

|    Model      |   Dataset    |   F-score   |
|:-------------:|:------------:|:-----------:|
|    ScispaCy   |     BC5CDR   |     84.1    |
|    SciBERT    |     BC5CDR   |     90.0    |
|    BioBERT    |     BC5CDR   |     90.3    |
|    SapBERT    |     BC5CDR-d |  **93.5**   |
|  *Our model*  |     BC5CDR   |     73.0    |

## :high_brightness: Acknowledgements
+ This work was funded by the Ministry of Science and Technology scholarship for STEM research students.

## :email: Contact
- rinagalperin@gmail.com
