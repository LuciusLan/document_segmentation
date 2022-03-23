TRAIN_PATH = './/train'
TEST_PATH = './/test'
TRAIN_LABEL = './/train.csv'
SUBMISSION = './/sample_submission.csv'
LABEL_2_ID = {'PAD':0, 'Claim': 1, 'Evidence': 2, 'Position': 3,
              'Concluding Statement': 4, 'Lead': 5, 'Counterclaim': 6, 'Rebuttal': 7, 'non': 8}
LABEL_BIO = {'PAD':0, 'B1': 1, 'I1': 2, 'B2': 3, 'I2': 4, 'B3': 5, 'I3': 6, 'B4': 7, 'I4': 8, 'B5': 9, 'I5': 10,
             'B6': 11, 'I6': 12, 'B7': 13, 'I7': 14, 'O': 15}
BOUNDARY_LABEL = {'PAD':0, 'B': 1, 'E': 2, 'O': 3}
BOUNDARY_LABEL_UNIDIRECTION = {'PAD':0, 'B': 1, 'O': 3}
TEST_SIZE = 0
DEV_SIZE = 0.1
MAX_LEN = 512
NUM_LABELS = 16
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCH = 5
LSTM_HIDDEN = 100
BIAFFINE_DROPOUT = 0.3
BASELINE = False
MODEL_CACHE_DIR = 'D:\\Dev\\roberta-base'
