from tqdm import tqdm

from transformer_model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###モデル学習と評価の関数定義###
def train(model, data, optimizer, criterion, PAD_IDX):
    model.train()
    losses = 0

    for src, tgt in data:
        src = torch.stack(src)  # バッチサイズx埋め込み次元数x時系列データの数
        tgt = tgt.clone().detach()  # 文章の単語数xバッチサイズ

        input_tgt = tgt[:-1, :]  # 最後の要素を削る
        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        optimizer.zero_grad()

        logit = model.forward(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )

        output_tgt = tgt[1:, :]
        loss = criterion(logit.reshape(-1, logit.shape[-1]), output_tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        return loss / len(data)


def evaluate(model, data, criterion, PAD_IDX):
    model.eval()
    losses = 0
    for src, tgt in data:
        src = torch.stack(src)  # バッチサイズx埋め込み次元数x時系列データの数
        tgt = tgt.clone().detach()  # 文章の単語数xバッチサイズ

        input_tgt = tgt[:-1, :]  # ここは変更すべき

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        logit = model.forward(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )
        output_tgt = tgt[1:, :]
        loss = criterion(logit.reshape(-1, logit.shape[-1]), output_tgt.reshape(-1))
        losses += loss.item()

    return losses / len(data)
