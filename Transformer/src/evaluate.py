import torch
import torch.nn.functional as F

from main import *
from preprocess import *
from transformer_model import generate_square_subsequent_mask

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def translate(model, seq, vocab_tgt, seq_len_tgt, START_IDX, END_IDX):
    model.eval()

    seq_len_src = seq.size()[0]
    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)

    # predicts = greedy_decode(
    #     model=model, src=seq,
    #     mask_src=mask_src, seq_len_tgt=seq_len_tgt,
    #     START_IDX=START_IDX, END_IDX=END_IDX
    # ).flatten()

    outputs, scores = beam_search(
        model=model, src=seq,
        mask_src=mask_src, seq_len_tgt=seq_len_tgt,
        START_IDX=START_IDX, END_IDX=END_IDX, beam_size=4
    )
    # print(np.array(predicts).shape)

    outputs, scores = zip(*sorted(zip(outputs, scores), key=lambda x: -x[1]))
    for o, output in enumerate(outputs):
        output_sentence = ' '.join(convert_indexes_to_text(output, vocab_tgt))
        print('out{}:{}'.format(o + 1, output_sentence))

    return None


def greedy_decode(model, src, mask_src, seq_len_tgt, START_IDX, END_IDX):
    src = src.to(device)  # 300x111
    mask_src = mask_src.to(device)

    memory = model.encode(src.unsqueeze(1), mask_src)  # 300x1x111
    ys = torch.ones(1, 1).fill_(START_IDX).type(torch.long).to(device)  # BOSから始まりますよ

    for i in range(seq_len_tgt - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)

        mask_tgt = (generate_square_subsequent_mask(ys.size(0), PAD_IDX).type(torch.bool)).to(device)

        # print('src:{}\nys:{}\nmemory:{}\nmask_tgt:{}'.format(src.shape, ys, memory.shape, mask_tgt))
        output = model.decode(ys, memory, mask_tgt)
        print("outputの形状", output.shape)
        output = output.transpose(0, 1)
        output = model.output(output[:, -1])
        print("model.outputのshape", output.shape)
        _, next_word = torch.max(output, dim=1)
        print(next_word)
        # print(_)
        # guess_words = torch.argsort(output, dim = 1)
        # next_word = guess_words.tolist().index(1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == END_IDX:
            break

    return ys


def beam_search(model, src, mask_src, seq_len_tgt, START_IDX, END_IDX, beam_size):
    # beam_sizeの定義どうなってるの？
    # なんで100でやった時できないのか？
    # 文章が長い方が尤度が高くなる傾向にあるのか？
    k = beam_size
    finished_beams = []
    finished_scores = []
    prev_probs = torch.zeros(k, 1, dtype=torch.float, device=device)  # 前の時刻の各ビームの大数尤度を保持しておく
    output_size = len(vocab_tgt)  # 語彙数
    # print(vocab_tgt)
    src = src.to(device)  # 300x111
    mask_src = mask_src.to(device)
    memory = model.encode(src.unsqueeze(1), mask_src).repeat(1, k, 1).contiguous()  # 300 x k x embedding_size
    # print("memoryのサイズ", memory.shape)
    # memory = model.encode(src.unsqueeze(1), mask_src)  # 300x1x111
    # print(memory.shape)
    ys = torch.ones(1, 1).fill_(START_IDX).type(torch.long).to(device)
    # beam_sizeの数だけrepeatする
    ys = ys.expand(1, k)  # 1xk
    ####

    # 各時刻ごとに処理
    for t in range(seq_len_tgt):
        # print(t + 1, "回目のループ")
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)

        mask_tgt = (generate_square_subsequent_mask(ys.size(0), PAD_IDX).type(torch.bool)).to(device)
        ####error####
        output = model.decode(ys, memory, mask_tgt)  # 1 x k x embedding_size
        # print(output.shape)
        # output = output.transpose(0, 1)
        output = model.output(output)  # 1 x k x output_size
        output_t = output[-1]  # k x output_size
        # print("outputのsize:", output.shape)
        # print("output_tのsize：", output_t.shape)

        log_probs = prev_probs + F.log_softmax(output_t, dim=-1)
        # print("log_probsのshape:", log_probs.shape)  # k x output_size
        scores = log_probs  # 対数尤度をスコアにする

        # スコアの高いビームとその単語を取得
        flat_scores = scores.view(-1)  # k*output_size

        if t == 0:
            flat_scores = flat_scores[:output_size]
            # prilat_scoresの形状：",flat_scores.shape)
        # スコアのトップk個の値(top_vs)とインデックス(top_is)を保持
        top_vs, top_is = flat_scores.data.topk(k)
        beam_indices = torch.div(top_is, output_size, rounding_mode='floor')
        word_indices = top_is % output_size

        # print("top_is:", top_is)
        # print("beam_indicies:", beam_indices)
        # print("word_indices:", word_indices)

        # ビームの更新
        _next_beam_indices = []
        _next_word_indices = []
        for b, w in zip(beam_indices, word_indices):
            if w.item() == END_IDX:
                k -= 1
                # print("終わりが来た時のysの形状：",ys.t().shape)
                # print("b：",b)
                # print("w：",w)
                beam = torch.cat([ys.t()[b], w.view(1, )])
                score = scores[b, w].item()
                finished_beams.append(beam)
                finished_scores.append(score)
                memory = model.encode(src.unsqueeze(1), mask_src).repeat(1, k, 1).contiguous()  # 300x1x111
            else:
                _next_beam_indices.append(b)
                _next_word_indices.append(w)
        if k == 0:
            break
        # print("_next_beam_indices : ", _next_beam_indices)
        next_beam_indices = torch.tensor(_next_beam_indices, device=device)
        # print("next_beam_indices : ", next_beam_indices)
        next_word_indices = torch.tensor(_next_word_indices, device=device)
        # print("next_word_indices : ", next_word_indices)

        # 次の時刻のDecoderの入力を更新
        ys = torch.index_select(
            ys, dim=-1, index=next_beam_indices
        )
        ys = torch.cat(
            [ys, next_word_indices.unsqueeze(0)], dim=0
        )

        # 各ビームの対数尤度を更新
        # print("log_probsの形状", log_probs.shape)
        flat_probs = log_probs.view(-1)
        # print("flat_probsの形状", flat_probs.shape) #tgt_vocab*2
        next_indices = (next_beam_indices + 1) * next_word_indices
        prev_probs = torch.index_select(
            flat_probs, dim=0, index=next_indices
        ).unsqueeze(1)
        # print("prev_probsのshape:", prev_probs.shape)
        # print("finished_beams:", finished_beams)
        # print("finished_scores:", finished_scores)
        # print("-" * 50)

    # 全てのビームが完了したらデータを整形
    outputs = [[idx.item() for idx in beam[1:-1]] for beam in finished_beams]

    return outputs, finished_scores
