import math

import torch.nn
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
import time
import numpy as np
from collections import OrderedDict

from train import train
from transformer_model import Seq2seqTransformer
from preprocess import *
from path_schema import *
from evaluate import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ファイルの読み込み
encoder_file_path = f"{DATA_PATH}/combined_beh.csv"
decoder_file_path = f"{DATA_PATH}/hmm_annotation_list.dat"
# encoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/combined_beh.csv"
# decoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/hmm_annotation_list.dat"

tokenizer_tgt = get_tokenizer("basic_english")  # 文章をすべて小文字にして単語ごとに分割する
vocab_tgt = build_vocab(read_texts(decoder_file_path), tokenizer_tgt)  # 単語辞書の作成

seq_src = read_seq(encoder_file_path)  # 入力時系列のリスト化# 1764x300x111
texts_tgt = read_texts(decoder_file_path)  # 文章をリスト化

batch_size = 200
PAD_IDX = vocab_tgt['<pad>']
START_IDX = vocab_tgt['<start>']
END_IDX = vocab_tgt['<end>']

vocab_size_tgt = len(vocab_tgt)
embedding_size = np.array(seq_src).shape[2]  # 111
nhead = 3  # 111の約数
dim_feedforward = 128
num_encoder_layer = 6
num_decoder_layer = 6
dropout = 0.1


def main():
    # 訓練データの構築
    train_data = data_preprocess(
        seq_src=seq_src, texts_tgt=texts_tgt, vocab_tgt=vocab_tgt, tokenizer_tgt=tokenizer_tgt
    )
    # print("train_data(エンコーダ側とデコーダ側の入力のデータの組)の要素数：", len(train_data))

    # ミニバッチを作る
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            collate_fn=generate_batch)  # シャッフルがFalseになっているから後で変える！
    # print("↓↓↓データローダーの第一要素↓↓↓\n", len(list(train_iter)[0][1][0]))  # 各列が文章に対応している. 今回の場合、(文章中の最大の単語数)x64x28 (64x28=1792)

    model = Seq2seqTransformer(
        num_encoder_layer, num_decoder_layer, embedding_size, vocab_size_tgt, dim_feedforward, dropout,
        nhead
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters())

    # 学習
    epoch = 100
    best_loss = float('-Inf')
    best_model = None
    patience = 50
    counter = 0

    for loop in range(1, epoch + 1):
        start_time = time.time()

        loss_train = train(
            model=model, data=train_iter, optimizer=optimizer, criterion=criterion, PAD_IDX=PAD_IDX
        )

        elapsed_time = time.time() - start_time

        # 検証データを作ってから
        # loss_valid=evaluate(
        #     model=model,data=valid_iter,criterion=criterion,PAD_IDX=PAD_IDX
        # )

        print('[{}/{}] train loss: {:.2f}  [{}{:.0f}s] count: {}, {}'.format(
            loop, epoch,
            loss_train,
            str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
            elapsed_time % 60,
            counter,
            '**' if best_loss > loss_train else ''
        ))

        best_model = model

        if counter > patience:
            break

        counter += 1

    # モデルの保存
    torch.save(best_model.state_dict(), f'{DATA_PATH}/translation_transformer.pth')
    return best_model


if __name__ == "__main__":
    # 学習済みモデルの読み込み
    best_model = Seq2seqTransformer(
        num_encoder_layer, num_decoder_layer, embedding_size, vocab_size_tgt, dim_feedforward, dropout,
        nhead
    ).to(device)
    best_model.load_state_dict(torch.load(f'{DATA_PATH}/translation_transformer.pth', map_location="cpu"))

    # best_model = main()
    mask_src = torch.zeros((300, 300), device=device).type(torch.bool)
    # seq = torch.tensor(seq_src[667])
    # print(seq.shape)
    seq2 = torch.tensor(np.array(np.random.randn(300, 111))).float()
    print(seq2.shape)
    predicted_sentence = translate(model=best_model, seq=seq2,
                                   vocab_tgt=vocab_tgt,
                                   seq_len_tgt=10,  # 最大系列長
                                   START_IDX=START_IDX, END_IDX=END_IDX)
    print(predicted_sentence)
