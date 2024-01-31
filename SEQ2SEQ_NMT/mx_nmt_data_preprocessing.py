import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, CuDNNGRU
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mark_start = 'ssss '
mark_end = ' eeee'
data_source = []
data_target = []
print("Reading lines...")
with open("space_cleaned_tr.txt", encoding='UTF-8') as tr_file:
    turkish_data = tr_file.readlines()
with open("space_cleaned_eng.txt", encoding='UTF-8') as eng_file:
    english_data = eng_file.readlines()
for tr_text, en_text in zip(turkish_data, english_data):
    tr_text = mark_start + tr_text.strip() + mark_end  # Türkçe cümlelerin başına ve sonuna işaretçileri ekle
    data_source.append(en_text)  # İngilizce cümleyi data_source listesine ekle
    data_target.append(tr_text)  # Türkçe cümleyi data_target listesine ekle
print("Tokenizer processing...")
class TokenizerWrap(Tokenizer):
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)

        self.fit_on_texts(texts)

        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)
        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
            truncating = 'post'
        self.num_tokens = [len(x) for x in self.tokens]
        self.max_tokens = np.mean(self.num_tokens) + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]
        text = " ".join(words)
        return text

    def text_to_tokens(self, text, reverse=False, padding=False):
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)
        if reverse:
            tokens = np.flip(tokens, axis=1)
            truncating = 'pre'
        else:
            truncating = 'post'
        tokens = pad_sequences(tokens,
                               maxlen=self.max_tokens,
                               padding='pre',
                               truncating=truncating)
        return tokens

tokenizer_source = TokenizerWrap(texts=data_source,
                                  padding='pre',
                                  reverse=True,
                                  num_words=None)

tokenizer_target = TokenizerWrap(texts=data_target,
                                  padding='post',
                                  reverse=False,
                                  num_words=None)

tokens_source = tokenizer_source.tokens_padded
tokens_target = tokenizer_target.tokens_padded

encoder_input_data = tokens_source
decoder_input_data = tokens_target[:, :-1]
decoder_output_data = tokens_target[:, 1:]

num_encoder_words = len(tokenizer_source.word_index)
num_decoder_words = len(tokenizer_target.word_index)
print("Data processing...")
embedding_size = 100
word2vec = {}

with open('glove.6B.100d.txt', encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

embedding_matrix = np.random.uniform(-1, 1, (num_encoder_words, embedding_size))
for word, i in tokenizer_source.word_index.items():
    if i < num_encoder_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

encoder_input = Input(shape=(None,), name='encoder_input')
encoder_embedding = Embedding(input_dim=num_encoder_words,
                              output_dim=embedding_size,
                              weights=[embedding_matrix],
                              trainable=True,
                              name='encoder_embedding')

state_size = 256

print("Model building...")
encoder_gru1 = GRU(state_size, name='encoder_gru1', return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2', return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3', return_sequences=False)

def connect_encoder():
    net = encoder_input
    net = encoder_embedding(net)
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)
    encoder_output = net
    return encoder_output

encoder_output = connect_encoder()

decoder_initial_state = Input(shape=(state_size,), name='decoder_initial_state')
decoder_input = Input(shape=(None,), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_decoder_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1', return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3', return_sequences=True)

decoder_dense = Dense(num_decoder_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(initial_state): 
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)
    return decoder_output

decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])

decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs=[decoder_output])
print("Model compiling...")
def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

decoder_target = tf.keras.Input(shape=(None, None), dtype=tf.int32)

model_train.compile(optimizer='rmsprop',
                    loss=sparse_cross_entropy)

path_checkpoint = 'checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, save_weights_only=True)

try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print('Checkpoint yüklenirken hata oluştu. Eğitime sıfırdan başlanıyor.')
    print(error)

x_data = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data}
y_data = {'decoder_output': decoder_output_data}
print("Model training...")
model_train.fit(x=x_data,
                y=y_data,
                batch_size=256,
                epochs=1,
                callbacks=[checkpoint])
