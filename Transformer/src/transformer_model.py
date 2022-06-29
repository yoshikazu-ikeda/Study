from pathlib import Path
import csv
import math

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import (
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from path_schema import DATA_PATH
from preprocess import (
    read_texts, read_seq, build_vocab, convert_text_to_indexes, convert_indexes_to_text, data_preprocess, generate_batch
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir_path = Path('model')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)

# encoder_file_path = f"{DATA_PATH}/combined_beh.csv"
# decoder_file_path = f"{DATA_PATH}/hmm_annotation_list.dat"

encoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/combined_beh.csv"
decoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/hmm_annotation_list.dat"

print(read_texts(decoder_file_path))

tokenizer_tgt = get_tokenizer("basic_english")  # 文章をすべて小文字にして単語ごとに分割する

vocab_tgt = build_vocab(read_texts(decoder_file_path), tokenizer_tgt)

# for word, index in vocab_tgt.items():  # 単語辞書の中身を見てみる
#     print('word: {: <8} -> Index: {: <2}'.format(word, index))

texts_tgt = read_texts(decoder_file_path)

train_data = data_preprocess(
    texts_tgt=texts_tgt, vocab_tgt=vocab_tgt, tokenizer_tgt=tokenizer_tgt
)

###確認###
print('インデックス化された文章')
# print(vocab_tgt)
print(train_data[77])
print(convert_indexes_to_text(train_data[77], vocab_tgt))



#ミニバッチを作る
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                        collate_fn=generate_batch)  # シャッフルがFalseになっているから後で変える！
print(list(train_iter)[0])  # 各列が文章に対応している. 今回の場合、(文章中の最大の単語数)x64x28 (64x28=1792)


###モデルの設計###
class Seq2seqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, embedding_size: int, seq_size_src: int,
                 vocab_size_tgt: int, dim_feedforward: int = 512, dropout: float = 0.1, nhead: int = 8):
        super(Seq2seqTransformer, self).__init__()

        # encoderの定義
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # decoderの定義
        self.token_embedding_tgt = TokenEmbedding(vocab_size_tgt, embedding_size)
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output = nn.Linear(embedding_size, vocab_size_tgt)


# Embedding層
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-torch.arrange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.embedding_pos[:token_embedding.size(0), :])


###マスキング###
def create_mask(tgt, PAD_IDX):
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt)

    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)
    return mask_tgt, padding_mask_tgt


def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask
