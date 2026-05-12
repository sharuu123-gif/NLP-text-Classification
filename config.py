import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, 'data')
RAW_DIR       = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SCRAPPED_DIR  = os.path.join(DATA_DIR, 'scrapped')
MODEL_DIR     = os.path.join(BASE_DIR, 'saved_models')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'outputs')
PLOT_DIR      = os.path.join(OUTPUT_DIR, 'plots')
REPORT_DIR    = os.path.join(OUTPUT_DIR, 'reports')

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_NAME  = "ag_news"
LABEL_NAMES   = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}
NUM_CLASSES   = 4

# ── Preprocessing ──────────────────────────────────────────────────────────────
MAX_LEN       = 128

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 3
LEARNING_RATE = 2e-5
TEST_SIZE     = 0.2
RANDOM_SEED   = 42

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_TYPE      = 'bert'          # options: 'lstm', 'cnn', 'bert'
BERT_MODEL_NAME = 'bert-base-uncased'

# LSTM / CNN settings
VOCAB_SIZE  = 50000
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_FILTERS = 128
