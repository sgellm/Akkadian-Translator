from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import pickle

combined_train = []
with open('model_training_single_tokenizer.pkl', 'rb') as f:
    combined_train = pickle.load(f)

combined_tokenizer = ByteLevelBPETokenizer()

combined_tokenizer.train_from_iterator(
    combined_train,
    vocab_size=24000,
    min_frequency=3,
    special_tokens=["<pad>", "<eos>", "<sos>", "<unk>", "<mask>"]
)

combined_tokenizer.enable_padding(
    pad_id=combined_tokenizer.token_to_id("<pad>"),
    pad_token="<pad>"
)

combined_tokenizer.save('combined_tokenizer.json')

wrapped_combo_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="combined_tokenizer.json",
    bos_token="<sos>",
    eos_token="<eos>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>"
)