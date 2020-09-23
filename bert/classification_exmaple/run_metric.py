# Run metric
from bert.bert_code import run_classifier
from bert.dataloader.aclimdb import aclImdb
from bert.classification_exmaple.cfg_example import *
from bert.utilty.utilty import create_tokenizer_from_hub_module, model_fn_builder
import tensorflow as tf

train, test = aclImdb().get_data()
print(train.columns)

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(
    lambda x: run_classifier.InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this example
                                          text_a=x[DATA_COLUMN],
                                          text_b=None,
                                          label=x[LABEL_COLUMN]), axis=1)

# get bert_code tokenizer form hub model
tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)

test_InputExamples = test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                      text_a=x[DATA_COLUMN],
                                                                      text_b=None,
                                                                      label=x[LABEL_COLUMN]), axis=1)

test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

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
tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)

print(estimator.evaluate(input_fn=test_input_fn, steps=None, checkpoint_path=checkpoint_path))