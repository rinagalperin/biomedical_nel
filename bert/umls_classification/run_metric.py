import pprint
import tensorflow as tf

from bert.bert_code import run_classifier
from bert.dataloader.contextual_relevance import ContextualRelevance
from bert.umls_classification.cfg import *
from bert.utilty.utilty import create_tokenizer_from_hub_module, model_fn_builder


def run(checkpoint_path, data_flie_path):
    # get model (make sure to change checkpoint according to the model in the configurations file)
    _, test, false_negatives_test_set = ContextualRelevance(data_flie_path).get_data()
    # get bert_code tokenizer form hub model
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)

    test_InputExamples = test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                          text_a=x[DATA_COLUMN],
                                                                          text_b=x[ANSWER_COLUMN],
                                                                          label=x[LABEL_COLUMN]), axis=1)

    test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH,
                                                                tokenizer)
    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        bert_model_hub=BERT_MODEL_HUB)

    estimator = tf.compat.v1.estimator.Estimator(model_fn, params={"batch_size": BATCH_SIZE})
    metric_result = estimator.evaluate(input_fn=test_input_fn, steps=None, checkpoint_path=checkpoint_path)
    metric_result['false_negatives'] += false_negatives_test_set
    metric_result['recall'] = metric_result['true_positives'] / (metric_result['true_positives'] + metric_result['false_negatives'])
    metric_result['eval_accuracy'] = (metric_result['true_positives'] + metric_result['true_negatives']) / (metric_result['true_positives'] + metric_result['false_negatives'] + metric_result['true_negatives'] + metric_result['false_positives'])
    precision = metric_result['precision']
    recall = metric_result['recall']

    metric_result['F1'] = 2 * (precision * recall) / (precision + recall)
    return metric_result


def main():
    model_checkpoint = 1299
    data_flie_path = '../../training_data/json_files/contextual_relevance/eng/medmentions_1.json'
    checkpoint_path = 'E:/nlp_model/output_model_medmentions_1/model.ckpt-{}'.format(model_checkpoint)

    metrics = run(checkpoint_path, data_flie_path)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(metrics)


if __name__ == '__main__':
    main()
