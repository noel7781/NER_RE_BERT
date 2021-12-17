import transformers
import torch

MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 5
BASE_MODEL_PATH = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "model.bin"
TRAINING_FILE = ""
VALIDATING_FILE = ""
TESTING_FILE = ""
TOKENIZER = transformers.BertTokenizer.from_pretrained(BASE_MODEL_PATH, do_lower_case=True)

