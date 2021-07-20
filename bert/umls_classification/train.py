import tensorflow as tf
from datetime import datetime

from bert.dataloader.contextual_relevance import ContextualRelevance
from bert.umls_classification.cfg import *
from bert.bert_code import run_classifier
from bert.utilty.utilty import create_tokenizer_from_hub_module, model_fn_builder
tf.compat.v1.disable_v2_behavior()


def run_train(data_file_path, output_dir):
    print('***** Model output directory: {} *****'.format(output_dir))

    # get data from data loader
    train, _, _ = ContextualRelevance(data_file_path).get_data()
    print(train.columns)

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(
        lambda x: run_classifier.InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this example
                                              text_a=x[DATA_COLUMN],
                                              text_b=x[ANSWER_COLUMN],
                                              label=x[LABEL_COLUMN]), axis=1)

    # get bert_code tokenizer form hub model
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB, False)
    print(tokenizer.tokenize("מריצים אימון..."))

    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.compat.v1.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        bert_model_hub=BERT_MODEL_HUB)

    estimator = tf.compat.v1.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    print('Beginning Training!')
    current_time = datetime.now()

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)


def main():
    data_flie_path = '../../training_data/json_files/contextual_relevance/eng/medmentions_1.json'
    output_dir = 'E:/nlp_model/output_model_medmentions_1'
    run_train(data_flie_path, output_dir)


if __name__ == '__main__':
    main()
