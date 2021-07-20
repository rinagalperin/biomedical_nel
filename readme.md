# Hebrew Contextual Relevance Model For UMLS Entity Linking Using BERT

## :dart: Goal
Linking terms to Unified Medical Language System (UMLS) entities can benefit text-analytics methods and make information retrieval and analysis 
easier and more accurate. UMLS concepts are mostly labeled in English, therefore NEL is especially challenging when the source language is different (e.g., Hebrew). This requires a general solution that can be easily adapted to any source language.  We address contextual relevance as an important factor when performing NEL and isolate the language-dependant components required for the task.  We operate in a low resource  setting (zero-shot),  where the ontologies are large, there is a lack of descriptive text  defining most entities, and labeled data can only cover a small portion of the ontology. This means we do not rely on any entity description resources. 

## :gift: Our contribution
- [x] We provide a general framework for cross-lingual UMLS NEL that can easily be adapted to any source language. While we focus here on UMLS NEL, our solution can be applied to solve other NEL tasks.
- [x] We introduce a novel enrichment technique that can help boost NEL performance.
- [x] We demonstrate results of our approach on both Hebrew and English as the source languages and compare to state-of-the-art (SOTA) solutions which are language-specific rather than multilingual. We improve on SOTA results on the Hebrew Camoni corpus, achieving +8.87 F1 on average across the three communities in the dataset. We also achieve SOTA on English dataset MedMentions, +4.4 F1 over the latest best result.

## :bulb: Approach
Our proposed end-to-end NEL model consists of four consecutive stages: (1)  **multilingual UMLS mapping**; generate a list $U_L$ of medical concepts in language $L$ mapped to universal CUIs from $C$, (2) **candidate generation**; consider all spans as candidate mentions and compute vector representations for both mentions and concepts, (3) **high recall matching**; use a semantic similarity based score function to generate the top matching entities with high recall being the priority and (4) **contextual relevance modeling**; embed each candidate span in its context, defined by the tokens surrounding it in the original text, alongside the label of whether or not the given mention is a medical concept in this context. This is used as an input to the BERT model fine-tuning.

![alt text](http://url/to/pipeline.png)

## :bar_chart: Results
- Camoni corpus ([MDTEL](https://github.com/yonatanbitton/mdtel))

| WINDOW_SIZE |   Accuracy  |  Precision  |   Recall   | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |     91.2%   |     67.5%   |    85.6%   |        27       |        77       |       912      |       160      |     75.5%    |
|      3      |     89.7%   |     62.8%   |    86.6%   |        25       |        96       |       893      |       162      |     72.8%    |
|      4      |     89.5%   |     62.3%   |    85.6%   |        27       |        97       |       892      |       160      |     72.1%    |


- MedMentions

| WINDOW_SIZE |   Accuracy  |  Precision  |   Recall   | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |     91.2%   |     67.5%   |    85.6%   |        27       |        77       |       912      |       160      |     75.5%    |
|      3      |     89.7%   |     62.8%   |    86.6%   |        25       |        96       |       893      |       162      |     72.8%    |
|      4      |     89.5%   |     62.3%   |    85.6%   |        27       |        97       |       892      |       160      |     72.1%    |


- BC5CDR

| WINDOW_SIZE |   Accuracy  |  Precision  |   Recall   | False negatives | False positives | True negatives | True positives |  F1 measure  |
|:-----------:|:-----------:|:-----------:|:----------:|:---------------:|:---------------:|:--------------:|:--------------:|:------------:|
|      2      |     91.2%   |     67.5%   |    85.6%   |        27       |        77       |       912      |       160      |     75.5%    |
|      3      |     89.7%   |     62.8%   |    86.6%   |        25       |        96       |       893      |       162      |     72.8%    |
|      4      |     89.5%   |     62.3%   |    85.6%   |        27       |        97       |       892      |       160      |     72.1%    |

## :high_brightness: Acknowledgements
+ Thanks to the [Ministry of Science and Technology](https://www.gov.il/he/departments/ministry_of_science_and_technology) for supporting our work

## :email: Contact
- rinag@post.bgu.ac.il
