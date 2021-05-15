import tensorflow as tf
import numpy as np
import training_data.utils

from bert.bert_code import run_classifier
from bert.hrm_classification.cfg import *
from bert.utilty.utilty import model_fn_builder, create_tokenizer_from_hub_module


def get_prediction(in_sentences, tokenizer, estimator):
    labels = [0, 1]
    input_examples = [run_classifier.InputExample(guid="", text_a=x[0], text_b=None, label=0) for x in
                      in_sentences]  # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH,
                                                       is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn, checkpoint_path=checkpoint_path)
    return [(sentence, np.exp(prediction['probabilities']) / np.sum(np.exp(prediction['probabilities'])), labels[prediction['labels']]) for sentence, prediction in
            zip(in_sentences, predictions)]


def perform_inference(predictions):
    ans = []
    for p in predictions:
        window_result = is_umls_window(p)
        ans.append(window_result)
    return ans


def is_umls_window(prediction, th=0.38):
    if prediction[1][1] > th:
        return 1
    return 0


model_fn = model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    bert_model_hub=BERT_MODEL_HUB)


def run_prediction_on_sentence(sentence):
    sentence_windows = [" ".join(x) for x in training_data.utils.window(sentence, WINDOW_SIZE, by_words=True)]
    estimator = tf.compat.v1.estimator.Estimator(model_fn, params={"batch_size": len(sentence_windows)})
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)
    net_output = get_prediction(sentence_windows, tokenizer, estimator)
    predictions = perform_inference(net_output)
    print(list(zip(predictions, sentence_windows)))


sentence = 'שלום אני חולה סוכרת ורציתי לדעת מהם היתרונות של משאבת אינסולין אומניפוד ומהם חסרונותיה כמו כן רציתי לדעת האם יש סיכון בחיסון שפעת חזירים לחולי סוכרת תודה'
run_prediction_on_sentence(sentence)

sentence = 'ערב טוב מזה כחשבועיים אני חשה תופעה מוזרה בכפות הרגליים התופעה היא שלפתע אצבעות הרגליים נתפסות לי לזמן קצר וזה מלווה בכאב קראתי כי לסוכרתיים יש נטייה לפגיעה ברגלים האם לתופעה זאת יש קשר לסכרת או למערכת עיצבית ? מאוד מתריד שאני גם לא יודת למי לפנות תודה מראש בברכת חג שבועות שמח'
run_prediction_on_sentence(sentence)

# _, test = HRMData(data_flie_path).get_data()
# window = test[DATA_COLUMN][1]
# print(is_umls_window(window))

#_, test = HRMData(data_flie_path).get_data()
# data = test[DATA_COLUMN][:2].to_list()
# predictions = get_prediction(data)
# print(predictions)
