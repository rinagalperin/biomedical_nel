# Run metric
import pprint

from bert.bert_code import run_classifier
from bert.dataloader.contextual_relevance import HRMData
from bert.hrm_classification.cfg import *
from bert.utilty.utilty import create_tokenizer_from_hub_module, model_fn_builder
import tensorflow as tf


def run(checkpoint_path, data_flie_path):
    # get model (make sure to change checkpoint according to the model in the configurations file)
    _, test = HRMData(data_flie_path).get_data()
    #if not is_baseline_data:
        #test = test[test.Is_Expanded_Term == 1]

    # get bert_code tokenizer form hub model
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)

    test_InputExamples = test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                          text_a=x[DATA_COLUMN],
                                                                          text_b=None,
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
    metric_result['recall'] = metric_result['true_positives'] / (metric_result['true_positives'] + metric_result['false_negatives'])
    metric_result['eval_accuracy'] = (metric_result['true_positives'] + metric_result['true_negatives']) / (metric_result['true_positives'] + metric_result['false_negatives'] + metric_result['true_negatives'] + metric_result['false_positives'])
    precision = metric_result['precision']
    recall = metric_result['recall']

    metric_result['F1'] = 2 * (precision * recall) / (precision + recall)
    return metric_result


def main():
    community = 'diabetes'
    data_flie_path = '../../training_data/training_data_hrm_{}_{}.json'.format(community, WINDOW_SIZE)

    model_checkpoint = 279
    checkpoint_path = 'E:/nlp_model/output_hrm_model_{}_{}/model.ckpt-{}'.format(community, WINDOW_SIZE, model_checkpoint)

    result = run(checkpoint_path, data_flie_path)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)


if __name__ == '__main__':
    main()
