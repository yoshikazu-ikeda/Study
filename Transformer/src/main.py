from transformer_model import Seq2seqTransformer
from preprocess import *


def main():
    #ファイルの読み込み
    encoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/combined_beh.csv"
    decoder_file_path = "//fukuoka/share/YoshikazuIkeda/Transformer/data/capdata/hmm_annotation_list.dat"

    vocab_tgt = build_vocab(read_texts(decoder_file_path), tokenizer_tgt)

    vocab_size_tgt = len(vocab_tgt)
    embedding_size = 200
    nhead = 8
    dim_feedforward = 100
    num_encoder_layer = 2
    num_decoder_layer = 2
    dropout = 0.1

    model = Seq2seqTransformer(
        num_encoder_layer, num_decoder_layer, embedding_size, vocab_size_tgt, dim_feedforward, dropout, nhead
    )


if __name__ == "__main__":
    main()
