import numpy
import keras
from keras.models import Model
from keras import layers
from keras.layers import \
    Input, Embedding, Lambda, Bidirectional, LSTM, \
    Conv1D, GlobalMaxPooling1D, Dot, BatchNormalization, Activation
from keras.preprocessing.sequence import pad_sequences
import dprocess

max_len = 64
embedding_dim = 300
lstm_unit = 512
conv_unit = 64
conv_step = 5
conv_layer_num = 3

epoches = 8
batch_size = 128

train_path = 'BoP2017-DBQA.train.txt'
dev_path = 'BoP2017-DBQA.dev.txt'
vec_path = 'zh.tsv'

dprocess.load_qa_list([train_path, dev_path])
embedding_matrix = dprocess.get_embedding_matrix(
    vec_path, embedding_dim
)

batch_per_epoch = len(dprocess.qa_list) // batch_size * 5

dev_q, dev_a, dev_tag = dprocess.get_dev_data(dev_path)

dev_q = pad_sequences(numpy.array(dev_q), max_len)
dev_a = pad_sequences(numpy.array(dev_a), max_len)
dev_tag = numpy.array(dev_tag)

input_q = Input((max_len,))
input_a1 = Input((max_len,))
input_a2 = Input((max_len,))

embedding_layer = Embedding(
    len(embedding_matrix), embedding_dim,
    weights=[embedding_matrix], input_length=max_len,
    trainable=False
)
emb_q = embedding_layer(input_q)
emb_a1 = embedding_layer(input_a1)
emb_a2 = embedding_layer(input_a2)

shared_lstm = Bidirectional(
    LSTM(lstm_unit, implementation=2, return_sequences=True)
)
encode_q = shared_lstm(emb_q)
encode_a1 = shared_lstm(emb_a1)
encode_a2 = shared_lstm(emb_a2)

for k in range(conv_layer_num):
    conv = Conv1D(
        conv_unit, conv_step,
        padding='valid', strides=1
    )
    relu = Activation('relu')
    batchnorm = BatchNormalization()


    def bn_conv(x):
        return relu(batchnorm(conv(x)))


    encode_q = bn_conv(encode_q)
    encode_a1 = bn_conv(encode_a1)
    encode_a2 = bn_conv(encode_a2)

vec_q = GlobalMaxPooling1D()(encode_q)
vec_a1 = GlobalMaxPooling1D()(encode_a1)
vec_a2 = GlobalMaxPooling1D()(encode_a2)

cosine_1 = Dot(axes=1, normalize=True)([vec_q, vec_a1])
cosine_2 = Dot(axes=1, normalize=True)([vec_q, vec_a2])

neg = Lambda(lambda x: -x, output_shape=lambda x: x)
# get_first = Lambda(lambda x: x[:, 0], output_shape = lambda x: x[: -1])
sub = layers.add([cosine_1, neg(cosine_2)])

model = Model(inputs=[input_q, input_a1, input_a2], outputs=[sub, cosine_1, cosine_2])
model.compile(
    optimizer='adam', loss='mean_squared_error', loss_weights=[0.6, 0.2, 0.2]
)


for k in range(epoches):
    print('Epoch: %d' % k)
    train_q, train_a1, train_a2, train_tag = \
        dprocess.get_train_samples(batch_size * batch_per_epoch, 0.25, 0.25)
    train_q = pad_sequences(numpy.array(train_q), max_len)
    train_a1 = pad_sequences(numpy.array(train_a1), max_len)
    train_a2 = pad_sequences(numpy.array(train_a2), max_len)
    train_tag = numpy.array(train_tag)
    train_t1, train_t2 = dprocess.trans_tag(train_tag)
    model.fit(
        [train_q, train_a1, train_a2], [train_tag, train_t1, train_t2],
        batch_size=batch_size, epochs=1,
    )
    preds = model.predict([dev_q, dev_a, dev_a], batch_size=batch_size)
    mrr = dprocess.evaluate_mrr(dev_tag, preds)
    print('\nmrr = ', mrr)


model.save('model1', overwrite=True)
dprocess.save_word_dict('word_dict1.txt')

