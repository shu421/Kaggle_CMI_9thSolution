def collate(inputs, labels=None):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    if labels:
        labels = labels[:, :mask_len]
        return inputs, labels
    else:
        return inputs
