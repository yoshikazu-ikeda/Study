import torch

from main import *
from preprocess import *
from transformer_model import generate_square_subsequent_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def translate(model, seq, mask_src, vocab_tgt, seq_len_tgt, START_IDX, END_IDX):
    model.eval()

    predicts = greedy_decode(
        model=model, src=seq,
        mask_src=mask_src, seq_len_tgt=seq_len_tgt,
        START_IDX=START_IDX, END_IDX=END_IDX
    ).flatten()

    words = convert_indexes_to_text(predicts, vocab_tgt)

    return words


def greedy_decode(model, src, mask_src, seq_len_tgt, START_IDX, END_IDX):
    src = src.to(device)
    # memory = model.encode(src, None)
    memory = model.transformer_encoder(model.positional_encoding(src.unsqueeze(-2)), None)
    ys = torch.ones(1, 1).fill_(START_IDX).type(torch.long).to(device)

    for i in range(seq_len_tgt - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)

        mask_tgt = (generate_square_subsequent_mask(ys.size(0), PAD_IDX).type(torch.bool)).to(device)

        # print('src:{}\nys:{}\nmemory:{}\nmask_tgt:{}'.format(src.shape, ys, memory.shape, mask_tgt))
        output = model.decode(ys, memory, mask_tgt)
        output = output.transpose(0, 1)
        output = model.output(output[:, -1])
        _, next_word = torch.max(output, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == END_IDX:
            break

    return ys
