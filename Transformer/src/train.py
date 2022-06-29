from transformer_model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###モデル学習と評価の関数定義###
def train(model, data, optimizer, criterion, PAD_IDX):
    model.train()
    losses = 0
    for src, tgt in tqdm(data):
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]
        mask_tgt, padding_mask_tgt = create_mask(input_tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_tgt=mask_tgt,
            padding_mask_tgt=padding_mask_tgt
        )

        optimizer.zero_grad()

        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        return loss / len(data)


def evaluate(model, data, criterion, PAD_IDX):
    model.eval()
    losses = 0
    for src, tgt in data:
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, 1]  # ここは変更すべき

        mask_tgt, padding_mask_tgt = create_mask(tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_tgt=mask_tgt,
            padding_mask_tgt=padding_mask_tgt
        )
        output_tgt=tgt[-1:,:]
        loss=criterion(logits.reshape(-1,logits.shape[-1]),output_tgt.reshape(-1))
        losses+=loss.item()

    return losses/len(data)
