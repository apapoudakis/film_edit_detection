
def accuracy(outputs, targets):
    _, pred = outputs.topk(k=1, dim=1)
    num_correct = (pred == targets).float().sum()
    acc = (num_correct/targets.shape[0]).item()
    return acc
