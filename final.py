import dprocess
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy

max_len = 64
batch_size = 128

test_path = 'BoP2017-DBQA.test.txt'
vec_path = 'zh.tsv'

dprocess.load_word_dict('word_dict1.txt')

test_q, test_a = dprocess.get_test_data(test_path)

test_q = pad_sequences(numpy.array(test_q), max_len)
test_a = pad_sequences(numpy.array(test_a), max_len)


model = keras.models.load_model('model1')

preds = model.predict([test_q, test_a, test_a], batch_size=batch_size)


preds = [t[0] for t in preds[1]]

pred_dev = open('pred_test.txt', 'w')
for i in range(len(preds)):
    pred_dev.write(str(preds[i]) + '\n')

pred_dev.close()

