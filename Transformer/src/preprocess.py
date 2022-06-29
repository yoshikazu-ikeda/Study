import csv
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence




def read_texts(file_path):  # デコーダーの入力となる文章をリスト化する関数
    texts = []
    with open(file_path, 'r') as file:
        for row in csv.reader(file, delimiter='\t'):
            texts.append(row[1])
    return texts


# print(read_texts(decoder_file_path))

def read_seq(file_path):  # エンコーダーの入力となる時系列データをリスト化する関数(3次元リスト)
    sequences = []
    sequence = []
    with open(file_path, 'r') as file:
        for row in csv.reader(file, delimiter='\t'):
            if row[0] == '0':
                sequences.append(sequence)
                sequence = [list(row[1:112])]
            else:
                sequence.append(list(row[1:112]))
        sequences.pop(0)
    return sequences


def build_vocab(texts, tokenizer):  # 単語辞書の作成
    counter = Counter()
    specials = {'<unk>': 0, '<pad>': 1, '<start>': 2, '<end>': 3}
    for text in texts:
        for word in tokenizer(text):
            counter[word] += 1
    keys = list(counter.keys())
    for i in range(len(counter)):
        counter[keys[i]] = i + 4
    counter_dict = dict(counter)
    specials.update(counter_dict)
    return specials


# 単語をインデックスに変換
def convert_text_to_indexes(text, vocab, tokenizer):
    return [vocab['<start>']] + [vocab[token] for token in tokenizer(text.strip("\n"))] + [vocab['<end>']]


# インデックスを単語に戻す
def convert_indexes_to_text(indexes, vocab):
    vocab_swap = {v: k for k, v in vocab.items()}  # 単語辞書のキーとバリューを交換
    return [vocab_swap[int(index)] for index in indexes]


# デコーダの入力になる単語をインデックスに変換
def data_preprocess(texts_tgt, vocab_tgt, tokenizer_tgt):
    data = []
    for tgt in texts_tgt:
        tgt_tensor = torch.tensor(
            convert_text_to_indexes(text=tgt, vocab=vocab_tgt, tokenizer=tokenizer_tgt),
            dtype=torch.long
        )
        data.append(tgt_tensor)

    return data


def generate_batch(data_batch):  # ミニバッチの作成
    batch_tgt = []
    for tgt in data_batch:
        batch_tgt.append(tgt)

    batch_tgt = pad_sequence(batch_tgt, padding_value=PAD_IDX)  # 短い文章に対してパディングする

    return batch_tgt
