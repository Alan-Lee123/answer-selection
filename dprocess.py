# -*- coding:utf-8 -*-
import re
import random
import numpy
import jieba

word_dict = {}
qa_list = []
punc_list = [
    '、', '，', '。', '！', '？',
    '（', '）', '【', '】', '《', '》', '‘', '’', '“', '”',
    ',', '.', '!', '?', '(', ')', '[', ']', '{', '}', '\'', '\"'
]


def sentence_to_words(sentence):
    words = []
    items = sentence.split()
    for item in items:
        item_words = list(jieba.cut(item))
        hint_word = 0
        for i in range(len(item_words)):
            if (i + 2) < len(item_words) and item_words[i] == '[' and item_words[i + 2] == ']':
                if item_words[i + 1].isdigit():
                    hint_word = 1
                    continue
            if hint_word != 0:
                hint_word = (hint_word + 1) % 3
                continue
            if item_words[i] in punc_list:
                continue
            words.append(item_words[i])
    return words


def words_to_indexes(items):
    for i in range(len(items)):
        if items[i] in word_dict:
            items[i] = word_dict[items[i]]
        else:
            items[i] = 0


def update_tokens(tokens, words):
    for word in words:
        if word in tokens:
            tokens[word] += 1
        else:
            tokens.setdefault(word, 1)


def load_qa_list(filenames):
    qa_table = {}
    for filename in filenames:
        infile = open(filename, 'r', encoding='utf-8')
        for line in infile:
            items = line.split('\t')
            if items[1] not in qa_table:
                qa_table.setdefault(items[1], [[], []])
            qa_table[items[1]][items[0] != '0'].append(items[2])
        infile.close()

    for qu in qa_table:
        qa_list.append([qu, qa_table[qu][0], qa_table[qu][1]])

    tokens = {}
    for i in range(len(qa_list)):
        qa_list[i][0] = sentence_to_words(qa_list[i][0])
        update_tokens(tokens, qa_list[i][0])
        for j in range(len(qa_list[i][1])):
            qa_list[i][1][j] = sentence_to_words(qa_list[i][1][j])
            update_tokens(tokens, qa_list[i][1][j])
        for j in range(len(qa_list[i][2])):
            qa_list[i][2][j] = sentence_to_words(qa_list[i][2][j])
            update_tokens(tokens, qa_list[i][2][j])

    sorted_tokens = sorted(
        list(tokens.items()),
        key=lambda it: it[1], reverse=True
    )
    for i in range(len(sorted_tokens)):
        word_dict.setdefault(sorted_tokens[i][0], i + 1)

    for i in range(len(qa_list)):
        words_to_indexes(qa_list[i][0])
        for j in range(len(qa_list[i][1])):
            words_to_indexes(qa_list[i][1][j])
        for j in range(len(qa_list[i][2])):
            words_to_indexes(qa_list[i][2][j])


def to_float_array(items):
    num_arr = []
    for item in items:
        if item != '':
            num_arr.append(float(item))
    return num_arr


def get_embedding_matrix(filename, embedding_dim):
    embedding_dict = {}
    word = ''
    infile = open(filename, 'r', encoding='utf-8')
    for line in infile:
        items = re.split(' |\t|\v|\r|\n|\f|\[|\]', line)
        if line[0] != ' ':
            word = items[1]
            if word not in word_dict:
                num_words = len(word_dict)
                word_dict.setdefault(word, num_words + 1)
            embedding_dict.setdefault(word, to_float_array(items[2:]))
        else:
            embedding_dict[word] += to_float_array(items)
    infile.close()
    embedding_matrix = numpy.zeros(
        (len(word_dict) + 1, embedding_dim), dtype='float32'
    )
    for word, index in word_dict.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is None:
            embedding_vector = [random.uniform(-1, 1) for i in range(300)]
            embedding_dict.setdefault(word, embedding_vector)
        if len(embedding_vector) != embedding_dim:
            print('Not padding embedding vector!')
        embedding_matrix[index] = numpy.asarray(
            embedding_vector, dtype='float32'
        )
    return embedding_matrix


def get_train_samples(num_samples, ratio_pn=0.33, ratio_np=0.33):
    if ratio_pn < 0 or ratio_np < 0 or ratio_pn + ratio_np > 1:
        print('illegal ratios!')
        return []
    train_samples = []
    num_pn_samples = int(num_samples * ratio_pn)
    num_np_samples = int(num_samples * ratio_np)
    num_0_samples = num_samples - num_pn_samples - num_np_samples
    for i in range(num_pn_samples):
        idx_q = random.randint(0, len(qa_list) - 1)
        while (len(qa_list[idx_q][1]) == 0 or len(qa_list[idx_q][2]) == 0):
            idx_q = random.randint(0, len(qa_list) - 1)
        idx_pa = random.randint(0, len(qa_list[idx_q][2]) - 1)
        idx_na = random.randint(0, len(qa_list[idx_q][1]) - 1)
        train_samples.append(
            (
                1, qa_list[idx_q][0],
                qa_list[idx_q][2][idx_pa], qa_list[idx_q][1][idx_na]
            )
        )
    for i in range(num_np_samples):
        idx_q = random.randint(0, len(qa_list) - 1)
        while (len(qa_list[idx_q][1]) == 0 or len(qa_list[idx_q][2]) == 0):
            idx_q = random.randint(0, len(qa_list) - 1)
        idx_na = random.randint(0, len(qa_list[idx_q][1]) - 1)
        idx_pa = random.randint(0, len(qa_list[idx_q][2]) - 1)
        train_samples.append(
            (
                -1, qa_list[idx_q][0],
                qa_list[idx_q][1][idx_na], qa_list[idx_q][2][idx_pa]
            )
        )
    for i in range(num_0_samples):
        idx_q = random.randint(0, len(qa_list) - 1)
        while (len(qa_list[idx_q][1]) < 2):
            idx_q = random.randint(0, len(qa_list) - 1)
        idx_na1 = random.randint(0, len(qa_list[idx_q][1]) - 1)
        idx_na2 = random.randint(0, len(qa_list[idx_q][1]) - 1)
        while (idx_na1 == idx_na2):
            idx_na2 = random.randint(0, len(qa_list[idx_q][1]) - 1)
        train_samples.append(
            (
                0, qa_list[idx_q][0],
                qa_list[idx_q][1][idx_na1], qa_list[idx_q][1][idx_na2]
            )
        )
    random.shuffle(train_samples)
    train_q = [train_samples[i][1] for i in range(num_samples)]
    train_a1 = [train_samples[i][2] for i in range(num_samples)]
    train_a2 = [train_samples[i][3] for i in range(num_samples)]
    train_tag = [train_samples[i][0] for i in range(num_samples)]
    return train_q, train_a1, train_a2, train_tag


def get_dev_data(filename):
    infile = open(filename, 'r', encoding='utf-8')
    dev_q, dev_a, dev_tag = [], [], []
    last_q, last_a, last_tag = '', [], []
    for line in infile:
        items = line.split('\t')
        if items[1] != last_q:
            q = sentence_to_words(last_q)
            words_to_indexes(q)
            for a in last_a:
                dev_q.append(q)
                dev_a.append(sentence_to_words(a))
                words_to_indexes(dev_a[len(dev_a) - 1])
            if last_tag:
                dev_tag.append(last_tag)
            last_q, last_a, last_tag = items[1], [], []
        last_a.append(items[2])
        last_tag.append(int(items[0] != '0'))
    if last_tag:
        q = sentence_to_words(last_q)
        words_to_indexes(q)
        for a in last_a:
            dev_q.append(q)
            dev_a.append(sentence_to_words(a))
            words_to_indexes(dev_a[len(dev_a) - 1])
        dev_tag.append(last_tag)

    return dev_q, dev_a, dev_tag


def load_word_dict(filename):
    infile = open(filename, 'r', encoding='utf-8')
    line = infile.read()
    infile.close()
    words = line.split('\t')
    word_num = 0
    word_dict.clear()
    for word in words:
        word_num += 1
        word_dict.setdefault(word, word_num)


def save_word_dict(filename):
    sorted_tokens = sorted(
        list(word_dict.items()),
        key=lambda it: it[1], reverse=False
    )
    outfile = open(filename, 'w', encoding='utf-8')
    for token, index in sorted_tokens:
        outfile.write(token + '\t')
    outfile.close()


def get_test_data(filename):
    infile = open(filename, 'r', encoding='utf-8')
    test_q, test_a = [], []
    for line in infile:
        items = line.split('\t')
        test_q.append(sentence_to_words(items[0]))
        test_a.append(sentence_to_words(items[1]))
        words_to_indexes(test_q[len(test_q) - 1])
        words_to_indexes(test_a[len(test_a) - 1])
    return test_q, test_a


def trans_tag(tag):
    t1 = numpy.zeros(len(tag))
    t2 = numpy.zeros(len(tag))
    for k in range(len(tag)):
        if tag[k] == 0:
            t1[k] = 0
            t2[k] = 0
        elif tag[k] == 1:
            t1[k] = 1
            t2[k] = 0
        else:
            t1[k] = 0
            t2[k] = 1
    return t1, t2


def evaluate_mrr(tag_lists, preds):
    idx = 0
    cnt = 0
    score = 0
    preds = [t[0] for t in preds[1]]
    for tags in tag_lists:
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
    return score / cnt
