import torch


def get_answer_loss(operation, batch, model):
    #   operation: either "ga" (gradient ascent) or "gd" (gradient descent).

    device = model.device
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    # Logits shape is (batch size, sequence length, vocab size)
    shift_logits = outputs.logits[:, :-1, :]
    # Label shape is (batch size, sequence length)
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        # output has dimension of vocabulary size and label indicate the index of true token
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":
            position_loss = -position_loss

        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only consider answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()
    return final_loss


def compute_kl(pretrained_model, current_model, batch):
    device_0 = current_model.device
    normal_outputs = current_model(
        batch["input_ids"].to(device_0),
        attention_mask=batch["attention_mask"].to(device_0),
        labels=batch["labels"].to(device_0),
    )

    device_1 = pretrained_model.device
    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device_1),
            attention_mask=batch["attention_mask"].to(device_1),
            labels=batch["labels"].to(device_1),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1).to(device_0)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1).to(device_0)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss
