import json

import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, GRU, Input, Lambda, Multiply, Softmax, \
    TimeDistributed
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import tokenizer_from_json

tf.random.set_seed(54)  # for consistent reproduction

data_train = np.load('data/data_train.npy')
data_val = np.load('data/data_val.npy')
data_test = np.load('data/data_test.npy')
labels_categorical_train = np.load('data/labels_categorical_train.npy')
labels_categorical_val = np.load('data/labels_categorical_val.npy')
labels_categorical_test = np.load('data/labels_categorical_test.npy')
EMBEDDING_WEIGHTS = np.load('data/embedding_weights.npy')

NUM_CLASSES = labels_categorical_train.shape[1]
MAX_NUM_VOCAB, EMBEDDING_DIM = EMBEDDING_WEIGHTS.shape
MAX_NUM_SENTS, MAX_NUM_WORDS_IN_SENT = data_train.shape[1:]
GRU_OUTPUT_DIM = 50
HIDDEN_STATES_DIM = GRU_OUTPUT_DIM * 2

print('training   data and labels:', data_train.shape, labels_categorical_train.shape)
print('validation data and labels:', data_val.shape, labels_categorical_val.shape)
print('test       data and labels:', data_test.shape, labels_categorical_test.shape)
print('MAX_NUM_VOCAB             :', MAX_NUM_VOCAB)
print('EMBEDDING_DIM             :', EMBEDDING_DIM)
print('MAX_NUM_SENTS             :', MAX_NUM_SENTS)
print('MAX_NUM_WORDS_IN_SENT     :', MAX_NUM_WORDS_IN_SENT)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:  # memory growth needs to be the same across GPUs
        tf.config.experimental.set_memory_growth(gpu, True)


def word_encoder(name):
    # ----- word encoder -----
    sentence_input = Input(shape=(MAX_NUM_WORDS_IN_SENT,), name='encoded_sentence')
    words_vectors = Embedding(input_dim=MAX_NUM_VOCAB, output_dim=EMBEDDING_DIM, weights=[EMBEDDING_WEIGHTS],
                              input_length=MAX_NUM_WORDS_IN_SENT,
                              mask_zero=False, trainable=True, name='words_vectors')(sentence_input)
    h_it = Bidirectional(GRU(GRU_OUTPUT_DIM, return_sequences=True), name='words_annotations')(
        words_vectors)  # shape: [botch_size, num_words_in_sent, hidden_states_dim]
    # ----- word_attention -----
    u_it = Dense(HIDDEN_STATES_DIM, activation='tanh', name='hidden_representation_of_words_annotations')(
        h_it)  # shape: [batch_size, num_words_in_sent, hidden_states_dim]
    alpha_it = Softmax(axis=1, name='normalized_words_importance_weights')(
        Dense(1, use_bias=False, name='context_vector')(u_it))  # shape: [batch_size, num_words_in_sent, 1]
    s_i = Lambda(lambda x: K.sum(x, axis=1), name='sum_of_weighted_words_annotations')(
        Multiply(name='weighted_words_annotations')([alpha_it, h_it]))  # shape: [batch_size, hidden_states_dim]
    return Model(sentence_input, s_i, name=name)


test = word_encoder('word_encoder')
test.summary(line_length=160)


def build_model():
    # -----= sentence encoder -----
    sentences_input = Input(shape=(MAX_NUM_SENTS, MAX_NUM_WORDS_IN_SENT), batch_size=None, dtype='int32',
                            name='encoded_sentences')
    words_attentions = TimeDistributed(word_encoder(name='word_encoders'), name='words_attentions')(sentences_input)
    h_i = Bidirectional(GRU(GRU_OUTPUT_DIM, return_sequences=True), name='sentences_annotations')(
        words_attentions)  # shape: [batch size, num_sents, hidden_dim_size]
    # ----- sentence attention -----
    u_i = Dense(HIDDEN_STATES_DIM, activation='tanh', name='hidden_representation_of_sentences_annotations')(
        h_i)  # shape: [batch_size, num_sents, hidden_states_dim]
    alpha_i = Softmax(axis=1, name='normalized_sentences_importance_weights')(
        Dense(1, use_bias=False, name='sentences_context_vector')(u_i))  # shape: [batch size, num_sents, 1]
    v = Lambda(lambda x: K.sum(x, axis=1), name='sum_of_weighted_sentences_annotations')(
        Multiply(name='weighted_sentences_annotations')([alpha_i, h_i]))  # shape: [batch_size, hidden_states_dim]
    # ----- document cLassification -----
    class_prediction = Dense(NUM_CLASSES, activation='softmax', name='classifier')(v)
    model = Model(sentences_input, class_prediction, name='hierarchical_attention_network')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3, amsgrad=True),
                  metrics=['accuracy'])
    return model


hierarchical_attention_network = build_model()
hierarchical_attention_network.summary(line_length=180)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=2),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2),
    ModelCheckpoint('model.{epoch:02d}-{val_loss:.3f}.h5', monitor='val_accuracy', save_best_only=True),
    TensorBoard(log_dir='logs', update_freq='batch')
]

history = hierarchical_attention_network.fit(x=data_train, y=labels_categorical_train,
                                             validation_data=(data_val, labels_categorical_val), shuffle=True,
                                             batch_size=64, epochs=10, callbacks=callbacks, verbose=1)

test_loss, test_accuracy = hierarchical_attention_network.evaluate(data_test, labels_categorical_test, verbose=0)
print('test loss: {:.3f}, test accuracy: {:.3f}'.format(test_loss, test_accuracy))

hierarchical_attention_network.save(f'model_{MAX_NUM_SENTS}_{MAX_NUM_WORDS_IN_SENT}.h5')

# evaluate model
model_loaded = load_model(f'model_{MAX_NUM_SENTS}_{MAX_NUM_WORDS_IN_SENT}.h5', custom_objects={'Functional': Model})
get_sentences_attentions = K.function(model_loaded.input,
                                      model_loaded.get_layer('normalized_sentences_importance_weights').output)
get_words_attentions = K.function(model_loaded.get_layer('words_attentions').layer.input,
                                  model_loaded.get_layer('words_attentions').layer.get_layer(
                                      'normalized_words_importance_weights').output)

print(get_sentences_attentions(data_train[0][np.newaxis, :])[0].shape, get_words_attentions(data_train[0]).shape)

document_index = np.random.randint(low=0, high=len(data_test) - 1)  # pick a random document index
document_encoded = data_test[document_index]

with open('data/tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(json.load(f))

word_attention_scale_factor = 1
sentence_attention_scale_factor = 3

# translate one-hot encoded (categorical) labels to one dimension
index_label_dict = np.load('data/index_label_dict.npy', allow_pickle=True)[()]
labels_categorical_pred = model_loaded.predict(data_test)
labels_pred = [index_label_dict[i] for i in labels_categorical_pred.argmax(axis=1)]
labels_test = [index_label_dict[i] for i in labels_categorical_test.argmax(axis=1)]
print(classification_report(y_true=labels_test, y_pred=labels_pred, zero_division=0))
