import csv
from collections import Counter

import numpy as np
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
    pad_list = [0] * 111
    max_seq = 0
    num_data = 0
    count = 0
    with open(file_path, 'r') as file:
        for row in csv.reader(file, delimiter='\t'):
            if int(row[0]) == 0:
                count = count + 1
                num_data += 1
                sequences.append(sequence)

                sequence = [list(float(row[i]) for i in range(1, 112))]
                if count == 20:
                    break
            else:
                if int(row[0]) > max_seq:
                    max_seq = int(row[0])
                sequence.append(list(float(row[i]) for i in range(1, 112)))
        sequences.append(sequence)  # 最後を付け足す
        sequences.pop(0)  # 一番初めの空のリストを削除

    # 最大の時系列データ数に併せて0パディングする
    for i in range(num_data):
        for j in range(max_seq - len(sequences[i]) + 1):
            if max_seq - len(sequences[i]) + 1 == 0:
                break
            sequences[i].append(pad_list)

    # new_sequences = np.array(sequences)[:, 0::10, :]

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
def data_preprocess(seq_src, texts_tgt, vocab_tgt, tokenizer_tgt):
    data = []
    for src, tgt in zip(seq_src, texts_tgt):
        src_tensor = torch.tensor(
            src, dtype=torch.float
        )
        tgt_tensor = torch.tensor(
            convert_text_to_indexes(text=tgt, vocab=vocab_tgt, tokenizer=tokenizer_tgt),
            dtype=torch.long
        )
        data.append((torch.t(src_tensor), tgt_tensor))
    return data


def generate_batch(data_batch):  # ミニバッチの作成
    batch_src = []
    batch_tgt = []
    for src, tgt in data_batch:
        batch_src.append(src)
        batch_tgt.append(tgt)

    batch_tgt = pad_sequence(batch_tgt, padding_value=1)  # 短い文章に対して<pad>でパディングする
    return batch_src, batch_tgt
