import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import (
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer
)
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq2seqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, embedding_size: int,
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

    def forward(self, src: Tensor, tgt: Tensor, mask_src: Tensor, mask_tgt: Tensor, padding_mask_src: Tensor,
                padding_mask_tgt: Tensor,
                memory_key_padding_mask: Tensor):
        embedding_src = self.positional_encoding(src.permute((2, 0, 1)))  # [300,128,111]
        memory = self.transformer_encoder(embedding_src, mask_src, src_key_padding_mask=padding_mask_src)
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(
            embedding_tgt, memory, mask_tgt, None, padding_mask_tgt, memory_key_padding_mask
        )
        return self.output(outs)

    def encode(self, src: Tensor, mask_src: Tensor):
        return self.transformer_encoder(self.positional_encoding(src), mask_src)

    def decode(self, tgt: Tensor, memory: Tensor, mask_tgt: Tensor):
        # print(self.positional_encoding(self.token_embedding_tgt(tgt)).shape)  # 1x1x111
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)


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

        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)  # 56
        # print("denのshape",den.shape)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)  # [5000,1]
        # print("posのshape",pos.shape)
        embedding_pos = torch.zeros((maxlen, embedding_size))  # [5000,111]
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den[:-1])  # [5000,111]
        embedding_pos = embedding_pos.unsqueeze(0)  # [1,5000,111]
        # print("embedding_posの形状：", embedding_pos.shape)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding: Tensor):
        # print("token_embedding", token_embedding.shape)  # [18,10,111]
        # print("embedding_posの形状", self.embedding_pos[:, :token_embedding.shape[1], :].shape)  # [1,10,111]
        return self.dropout(token_embedding + self.embedding_pos[:, :token_embedding.shape[1], :])


###マスキング###
def create_mask(src, tgt, PAD_IDX):  # tgt:input_tgt
    seq_len_src = src.size()[2]  # 300
    seq_len_tgt = tgt.size()[0]  # 9

    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)  # 300x300の全てFalseの行列
    mask_tgt = generate_square_subsequent_mask(seq_len_tgt, PAD_IDX)  # 上三角が全て-infでそれ以外が0の9x9行列

    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)  # PAD_IDXの場所だけをTrueにする,ここ怪しいかも
    padding_mask_src = torch.zeros(src.size()[0], src.size()[2], dtype=torch.bool)
    for i in range(src.size()[0]):
        for j in range(src.size()[2]):
            padding_mask_src[i, j] = (src[i, :, j].tolist() == [0] * 111)

    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt


def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask
