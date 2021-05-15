# Run metric
from bert.dataloader.contextual_relevance import ContextualRelevance
from bert.umls_classification.cfg import *
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_curve

# get bert_code tokenizer form hub model
_, test = ContextualRelevance(data_flie_path).get_data()


recall_label = np.ones(len(test[LABEL_COLUMN]))
label_list = test[LABEL_COLUMN]
precision, recall, _, _ = precision_recall_fscore_support(label_list, recall_label)
accuracy = accuracy_score(label_list, recall_label)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

true_pos, true_neg, false_pos, false_neg = perf_measure(label_list, recall_label)

x = {"eval_accuracy": accuracy,
# "f1_score": f1_score,
"precision": precision[1],
"recall": recall[1],
"true_positives": true_pos,
"true_negatives": true_neg,
"false_positives": false_pos,
"false_negatives": false_neg}
print(x)