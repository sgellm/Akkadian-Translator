from datasets import load_dataset
import re
from collections import defaultdict
import pickle

ds = load_dataset("cipher-ling/akkadian")
ds2 = load_dataset("hrabalm/mtm24-akkadian-v0")
ds3 = load_dataset("hrabalm/mtm24-akkadian-v1")
ds4 = load_dataset("hrabalm/mtm24-akkadian-v3")
ds5 = load_dataset("veezbo/akkadian_english_corpus")

def normalize(text):
    # remove '...', parentheses, and commas
    text = re.sub(r'\.\.\.|[(),]', '', text)
    # replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', '', text)
    return text.strip()

def enter_into_seqs(seqs, ak_norm, en_norm, key):
    if len(ak_norm) > 2 and len(en_norm) > 2:
        seqs[key].append(ak_norm + '\t' + en_norm)

ds_list = [ds2, ds3, ds4]

# Since the first dictionary has an extra key, we need to treat it differently

dict_list = ['train', 'test', 'validation']

seqs = defaultdict(list)

for key in dict_list:
    for entry in ds[key]['translation']: # list
        enter_into_seqs(seqs, normalize(entry['ak']), normalize(entry['en']), key)

dict_list.pop(2) # other dictionaries only have

for key in dict_list:
    for i, dataset in enumerate(ds_list):
        print(f'Beginning dataset {i+1}')
        for entry in dataset[key]:
            enter_into_seqs(seqs, normalize(entry['source']), normalize(entry['target']), key)

# in each section of seqs, there is a list of sentences that look like "[akkadian] + \t + [english]"
seq_set = list(set(seqs['train']))

# tokenizer data will be used to train the tokenizer
tokenizer_data = seq_set[:len(seq_set) - 75000]
combined_train = sorted([seq.replace('\t', ' ') for seq in tokenizer_data])

for line in ds5['train']['text']:
    line = re.sub(r'[^\w\s]', '', line)
    combined_train.append(line) # adding extra english to train the tokenizer

# model data will be used to train the model
model_data = seq_set[len(seq_set) - 75000:]

pickle.dump(combined_train, open('single_tokenizer_training_data.pkl', 'wb'))
pickle.dump(model_data, open('model_training_single_tokenizer.pkl', 'wb'))
pickle.dump(seqs, open('ak_en_seqs.pkl', 'wb'))