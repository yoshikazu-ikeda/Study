import math

import torch.nn
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
import time
import numpy as np

from train import train
from transformer_model import Seq2seqTransformer
from preprocess import *
from path_schema import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ファイルの読み込み
encoder_file_path = f"{DATA_PATH}/combined_beh.csv"
decoder_file_path = f"{DATA_PATH}/hmm_annotation_list.dat"
# encoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/combined_beh.csv"
# decoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/hmm_annotation_list.dat"

tokenizer_tgt = get_tokenizer("basic_english")  # 文章をすべて小文字にして単語ごとに分割する
vocab_tgt = build_vocab(read_texts(decoder_file_path), tokenizer_tgt)  # 単語辞書の作成

seq_src = read_seq(encoder_file_path)  # 入力時系列のリスト化
texts_tgt = read_texts(decoder_file_path)  # 文章をリスト化


# ハイパーパラメータの設定
batch_size = 128
PAD_IDX = vocab_tgt['<pad>']
START_IDX = vocab_tgt['<start>']
END_IDX = vocab_tgt['<end>']


def main():
    vocab_size_tgt = len(vocab_tgt)
    embedding_size = 200
    nhead = 8
    dim_feedforward = 100
    num_encoder_layer = 2
    num_decoder_layer = 2
    dropout = 0.1

    # 訓練データの構築
    train_data = data_preprocess(
        texts_tgt=texts_tgt, vocab_tgt=vocab_tgt, tokenizer_tgt=tokenizer_tgt
    )
    # ミニバッチを作る
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                            collate_fn=generate_batch)  # シャッフルがFalseになっているから後で変える！
    # print(list(train_iter)[0])  # 各列が文章に対応している. 今回の場合、(文章中の最大の単語数)x64x28 (64x28=1792)

    model = Seq2seqTransformer(
        num_encoder_layer, num_decoder_layer, embedding_size, vocab_size_tgt, dim_feedforward, dropout, nhead
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters())

    # 学習
    epoch = 100
    best_loss = float('Inf')
    best_model = None
    patience = 10
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

        if counter > patience:
            break

        counter += 1


if __name__ == "__main__":
    main()
