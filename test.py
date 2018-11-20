import dprocess
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy

max_len = 64
batch_size = 128

dev_path = 'BoP2017-DBQA.dev.txt'

dprocess.load_word_dict('word_dict1.txt')

dev_q, dev_a, dev_tag = dprocess.get_dev_data(dev_path)

dev_q = pad_sequences(numpy.array(dev_q), max_len)
dev_a = pad_sequences(numpy.array(dev_a), max_len)
dev_tag = numpy.array(dev_tag)

all_tags = []
index = []
for k in range(len(dev_tag)):
    index += [k] * len(dev_tag[k])
    all_tags += dev_tag[k]

model = keras.models.load_model('model1')

preds = model.predict([dev_q, dev_a, dev_a], batch_size=batch_size)

mrr = dprocess.evaluate_mrr(dev_tag, preds)
print("mrr = ", mrr)

preds = [t[0] for t in preds[1]]

pred_dev = open('pred_dev.txt', 'w')
for i in range(len(preds)):
    pred_dev.write(str(int(index[i])) + ', ' + str(all_tags[i]) + ', ' + str(preds[i]) + '\n')

pred_dev.close()

scores = open('score.txt', 'w')

idx = 0
cnt = 0
score = 0
for tags in dev_tag:
    pred = preds[idx: idx + len(tags)]
    idx += len(tags)
    p = numpy.argsort(pred)
    orders = numpy.zeros(len(p))
    for k in range(len(p)):
        orders[p[k]] = len(p) - k
    for k in range(len(tags)):
        if tags[k] == 1:
            cnt += 1
            score += 1. / orders[k]
            scores.write(str(int(orders[k])) + "\t" + str(score / cnt) + '\n')
scores.close()
