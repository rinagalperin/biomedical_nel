DATA_COLUMN = 'windows'
LABEL_COLUMN = 'labels'
COMMUNITIES = ['diabetes']

# manually change the following parameters to perform inference or to run metrics on a specific model:
is_baseline = False
WINDOW_SIZE = 5
community = 'diabetes'
model_checkpoint = 1974

data_flie_path = '../../training_data/training_data_hrm_{}_{}.json'.format(community, WINDOW_SIZE)
checkpoint_path = 'E:/nlp_model/output_hrm_model_{}_{}/model.ckpt-{}'.format(community, WINDOW_SIZE, model_checkpoint)

# label classes:
label_list = [0, 1]
#OUTPUT_DIR = '../output_model_{}'.format(WINDOW_SIZE)  # @param {type:"string"}
BERT_MODEL_HUB = "E:/nlp_model/pre-trained-model"
# BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'


BATCH_SIZE = 32
LEARNING_RATE = 2e-6
NUM_TRAIN_EPOCHS = 5.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
# Compute # train and warmup steps from batch size
num_train_steps = 0
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# utilty'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128

