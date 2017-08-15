import numpy
import dprocess
from cntk import input_variable, cross_entropy_with_softmax, \
        classification_error, sequence, Trainer, \
        cosine_distance, Axis, Value
from cntk.layers import Embedding, LSTM, Recurrence, Convolution, BatchNormalization
from cntk.layers.typing import tensor, Sequence
from cntk.ops import splice, relu, element_times, reduce_sum, sqrt, minus, one_hot
from cntk.learners import adam, UnitType, momentum_schedule, learning_rate_schedule
from cntk.logging import ProgressPrinter
from cntk.io import MinibatchSourceFromData

embedding_dim = 300
lstm_unit = 512
conv_unit = 64
conv_step = 5
conv_layer_num = 3

epoches = 8
batch_size = 8

loss_weights = [0.6, 0.2, 0.2]

train_path = 'BoP2017-DBQA.train.txt'
dev_path = 'BoP2017-DBQA.dev.txt'
pre_trained_word_vec_path = 'zh.tsv'

dprocess.load_qa_list([train_path, dev_path])
embedding_matrix = dprocess.get_embedding_matrix(
    pre_trained_word_vec_path, embedding_dim
)

batch_per_epoch = len(dprocess.qa_list) // batch_size * 5

dev_q, dev_a, dev_tag = dprocess.get_dev_data(dev_path)

input_q_axis = Axis('inputq')
input_a1_axis = Axis('inputa1')
input_a2_axis = Axis('inputa2')

input_q = sequence.input_variable(len(embedding_matrix), is_sparse=True, sequence_axis=input_q_axis, name='input_q')
input_a1 = sequence.input_variable(len(embedding_matrix), is_sparse=True, sequence_axis=input_a1_axis, name='input_a1')
input_a2 = sequence.input_variable(len(embedding_matrix), is_sparse=True, sequence_axis=input_a2_axis, name='input_a2')
input_tag = input_variable(shape=(3,), name='input_tag')

embedding_layer = Embedding(weights=embedding_matrix)

emb_q = embedding_layer(input_q)
emb_a1 = embedding_layer(input_a1)
emb_a2 = embedding_layer(input_a2)

lstm_fwd = Recurrence(LSTM(lstm_unit))
lstm_bwd = Recurrence(LSTM(lstm_unit), go_backwards=True)
bi_lstm = lambda t: Stabilizer()(splice(lstm_fwd(t), lstm_bwd(t)))

encode_q = bi_lstm(emb_q)
encode_a1 = bi_lstm(emb_a1)
encode_a2 = bi_lstm(emb_a2)

for k in range(conv_layer_num):
    conv = Convolution(
        conv_step, conv_unit,
        pad=True, sequential=True
    )

    relu = Activation('relu')
    batchnorm = BatchNormalization()

    def bn_conv(x):
        return relu(batchnorm(conv(x)))

    encode_q = bn_conv(encode_q)
    encode_a1 = bn_conv(encode_a1)
    encode_a2 = bn_conv(encode_a2)

encode_q = sequence.reduce_max(encode_q)
encode_a1 = sequence.reduce_max(encode_a1)
encode_a2 = sequence.reduce_max(encode_a2)

cosine_1 = cosine_distance(encode_q, encode_a1)
cosine_2 = cosine_distance(encode_q, encode_a2)

sub = minus(cosine_1, cosine_2)

r = splice(sub, cosine_1, cosine_2)

diff = minus(input_tag, r)

ce = reduce_sum(element_times(loss_weights, diff, diff), axis=0)

lr_s = learning_rate_schedule(lr=0.001, unit=UnitType.minibatch)
momentum_s = momentum_schedule(momentum=0.9)

progress_printer = ProgressPrinter(freq=100)

trainer = Trainer(r, ce, [adam(sub.parameters, lr=lr_s, momentum=momentum_s)], [progress_printer])

def evaluate(qs, anss, ts):
    cur = 0
    end = len(qs)
    preds = []
    while(cur < end):
        nt = cur + batch_size
        nt = min(nt, end)
        q = Value.one_hot(qs[cur:nt], len(embedding_matrix))
        a = Value.one_hot(anss[cur:nt], len(embedding_matrix))
        preds.extend(r.eval({input_q: q, input_a1: a, input_a2: a}))
        cur = nt
    mrr = dprocess.evaluate_mrr(dev_tag, preds)
    return mrr
    
for epoch in range(epoches):
    print('Epoch: %d' % epoch)

    for k in range(batch_per_epoch):
        train_q, train_a1, train_a2, train_tag = \
            dprocess.get_train_samples(batch_size, 0.25, 0.25)
        train_q = Value.one_hot(train_q, len(embedding_matrix))
        train_a1 = Value.one_hot(train_a1, len(embedding_matrix))
        train_a2 = Value.one_hot(train_a2, len(embedding_matrix))

        train_tags = dprocess.trans_tag(train_tag)
        params_dict = {input_q:train_q, input_a1:train_a1,
                            input_a2:train_a2, input_tag:train_tags}
        trainer.train_minibatch(params_dict)
    trainer.save_checkpoint('check_points/' + str(k) + '_checkpoint')
    mrr = evaluate(dev_q, dev_a, dev_tag)
    print('\nmrr = ', mrr)

r.save("cntk_model")
dprocess.save_word_dict('word_dict1.txt')

