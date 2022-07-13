import re
from ast import literal_eval
from metrics import precision, recall, f1_score


def check_overlap(begin1, end1, begin2, end2):
    return max(0, min(end1, end2) - max(begin1, begin2)) > 0


def gt_boundaries(gt_file):
    """
    Transform shots boundaries to type of cuts

    :param gt_file:
    :return:
    """

    with open(gt_file) as f:
        boundaries = f.readlines()
        boundaries = [re.sub('\s+', ' ', x.strip()) for x in boundaries]

    gt_cuts = []

    for i in range(len(boundaries)-1):
        begin1 = int(boundaries[i].split(" ")[0])
        end1 = int(boundaries[i].split(" ")[1])
        begin2 = int(boundaries[i+1].split(" ")[0])
        end2 = int(boundaries[i+1].split(" ")[1])

        if begin2-end1 == 1:
            gt_cuts.append((end1, begin2, 1))
        else:
            gt_cuts.append((end1, begin2, 2))

    return gt_cuts


def evaluate_predictions(pred_file, gt_file):
    with open(pred_file) as f:
        pred_cuts = f.readlines()
    pred_cuts = [literal_eval(x.strip()) for x in pred_cuts]

    gt_cuts = gt_boundaries(gt_file)
    print(pred_cuts)
    print(gt_cuts)

    num_preds = len(pred_cuts)
    total_gt_cuts = len(gt_cuts)
    correct_preds = 0

    for begin1, end1, pred in pred_cuts:
        for gt_begin2, gt_end2, label in gt_cuts:
            if check_overlap(begin1, end1, gt_begin2, gt_end2) and pred==label:
                correct_preds += 1

    prec = precision(correct_preds, num_preds)
    rec = recall(correct_preds, total_gt_cuts)
    f1 = f1_score(prec, rec)
    print(f"Precision: {prec}, Recall: {rec}, F1-score: {f1}")
