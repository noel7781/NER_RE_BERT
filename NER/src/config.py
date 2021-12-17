import transformers
import torch

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
BASE_MODEL_PATH = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "model.bin"
TRAINING_FILE = ""
TESTING_FILE = ""
TOKENIZER = transformers.BertTokenizer.from_pretrained(BASE_MODEL_PATH, do_lower_case=True)

