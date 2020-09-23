import tensorflow as tf
from bert.bert_code import run_classifier
from bert.classification_exmaple.cfg_example import *
from bert.utilty.utilty import model_fn_builder, create_tokenizer_from_hub_module


def get_prediction(in_sentences):
  labels = ["Negative", "Positive"]
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn, checkpoint_path=checkpoint_path)
  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]


model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  bert_model_hub=BERT_MODEL_HUB)

estimator = tf.compat.v1.estimator.Estimator(model_fn, params={"batch_size": BATCH_SIZE})
tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)


pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!"
]

predictions = get_prediction(pred_sentences)

print(predictions)