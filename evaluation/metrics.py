
def accuracy(outputs, targets):
    _, pred = outputs.topk(k=1, dim=1)
    num_correct = (pred == targets).float().sum()
    acc = (num_correct/targets.shape[0]).item()
    return acc


def precision(num_correct, num_preds):
    return num_correct/num_preds


def recall(num_correct, num_gt):
    return num_correct/num_gt


def f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)
