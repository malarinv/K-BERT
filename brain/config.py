import os


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

KGS = {
    'HowNet': os.path.join(FILE_DIR_PATH, 'kgs/HowNet.spo'),
    'HowNet_newt': os.path.join(FILE_DIR_PATH, 'kgs/HowNet_newt.spo'),
    'AIKG': os.path.join(FILE_DIR_PATH, 'kgs/AIKG.spo'),
    'AIKGTri': os.path.join(FILE_DIR_PATH, 'kgs/aikg_trigram.spo'),
    'Scholar': os.path.join(FILE_DIR_PATH, 'kgs/scholar_unigram.spo'),
    'CnDbpedia': os.path.join(FILE_DIR_PATH, 'kgs/CnDbpedia.spo'),
    'Medical': os.path.join(FILE_DIR_PATH, 'kgs/Medical.spo'),
}

MAX_ENTITIES = 2

# Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
ENT_TOKEN = '[ENT]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'

NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
]
