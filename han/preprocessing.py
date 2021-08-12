import json
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'entity_linker', 'textcat', 'entity_ruler'])
pd.set_option('display.max_colwidth', None)
tqdm.pandas()

# download movie review data from https://www.kaggle.com/c/word2vec-nlp-tutorial/data
df = pd.read_csv('data/labeledData.csv', dtype=str)

labels_categorical = pd.get_dummies(df['label'])
index_label_dict = {i: label for i, label in enumerate(labels_categorical.columns)}

MAX_NUM_VOCAB = 20000  # max num of vocabulary
tokenizer = Tokenizer(num_words=MAX_NUM_VOCAB, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                      split=' ', char_level=False, oov_token='<UNK>')
tokenizer.fit_on_texts(df['title'].values)
with open('data/tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)


def split_sentence(text):
    return [sentence.text.strip() for sentence in nlp(text).sents]


df['title_segmented'] = df['title'].progress_apply(lambda text: split_sentence(text))

num_sentences_in_doc_list = [len(sentence_list) for sentence_list in df['title_segmented']]
num_tokens_in_sentence_list = [len(sentence_tokenized) for sentence_tokenized in tokenizer.texts_to_sequences(
    [sentence for sentence_list in df['title_segmented'] for sentence in sentence_list])]

df_train_val, df_test, labels_categorical_train_val, labels_categorical_test = train_test_split(df, labels_categorical,
                                                                                                test_size=0.3,
                                                                                                stratify=labels_categorical,
                                                                                                random_state=42)

df_train, df_val, labels_categorical_train, labels_categorical_val = train_test_split(df_train_val,
                                                                                      labels_categorical_train_val,
                                                                                      test_size=0.1,
                                                                                      stratify=labels_categorical_train_val,
                                                                                      random_state=42)

MAX_NUM_SENTS = 80  # max num words in a sentence
MAX_NUM_WORDS_IN_SENT = 85  # max num sentences in a document


def generate_data(data_frame):
    data = np.zeros((len(data_frame['title']), MAX_NUM_SENTS, MAX_NUM_WORDS_IN_SENT), dtype=np.int32)
    for i, sentence_list in enumerate(tqdm(data_frame['title_segmented'])):
        for j, sentence_tokenized in enumerate(tokenizer.texts_to_sequences(sentence_list)):
            if j < MAX_NUM_SENTS:
                k = 0
                for word_index in sentence_tokenized:
                    if k < MAX_NUM_WORDS_IN_SENT:
                        data[i, j, k] = word_index
                        k += 1
    return data


data_train = generate_data(df_train)
data_val = generate_data(df_val)
data_test = generate_data(df_test)

np.save('data/data_train.npy', data_train)
np.save('data/data_val.npy', data_val)
np.save('data/data_test.npy', data_test)
np.save('data/labels_categorical_train.npy', labels_categorical_train.astype(np.float32))  # same dtype as model's
np.save('data/labels_categorical_val.npy', labels_categorical_val.astype(np.float32))
np.save('data/labels_categorical_test.npy', labels_categorical_test.astype(np.float32))
np.save('data/index_label_dict.npy', index_label_dict)

# download pre-trained word vectors from https://nlp.stanford.edu/projects/glove/
EMBEDDING_DIM = 200
GLOVE_EMBEDDING_PATH = f'data/glove.6B.{EMBEDDING_DIM}d.txt'

word_vector_dict = dict()
with open(GLOVE_EMBEDDING_PATH, encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        line_split = line.split()
        word_vector_dict[line_split[0]] = np.array(line_split[1:], dtype=np.float32)

# use mean vector as the unknown vector
unk_vector = np.mean(np.array(list(word_vector_dict.values()), dtype=np.float32), axis=0)
embedding_weights = np.zeros((MAX_NUM_VOCAB, EMBEDDING_DIM), dtype=np.float32)

num_success, num_failure = 0, 0
for (index, word) in tokenizer.index_word.items():
    if index < MAX_NUM_VOCAB:
        if word in word_vector_dict:
            embedding_weights[index] = word_vector_dict[word]
            num_success += 1
        else:
            embedding_weights[index] = unk_vector
            num_failure += 1
    else:
        break

print('word in pre-trained embedding:', num_success, '\nword not in pre-trained embedding:', num_failure)

np.save('data/embedding_weights.npy', embedding_weights)
