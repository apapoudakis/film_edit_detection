import os
import re
from ast import literal_eval
from metrics import precision, recall, f1_score
from models import deep_SBD
import inference
import utils
import pandas as pd
import torch
from collections import Counter
from operator import itemgetter


def check_overlap(begin1, end1, begin2, end2):
    return max(0, min(end1, end2) - max(begin1, begin2)) > 0


def segments_generator(frames, num_frames, overlap):
    for i in range(0, len(frames) - num_frames, overlap):
        yield frames[i:i + num_frames, :, :, :].permute(1, 0, 2, 3)


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

    for i in range(len(boundaries) - 1):
        begin1 = int(boundaries[i].split(" ")[0])
        end1 = int(boundaries[i].split(" ")[1])
        begin2 = int(boundaries[i + 1].split(" ")[0])
        end2 = int(boundaries[i + 1].split(" ")[1])

        if begin2 - end1 == 1:
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
    hard_preds = Counter(map(itemgetter(2), pred_cuts))[1]
    gradual_preds = Counter(map(itemgetter(2), pred_cuts))[2]

    total_gt_cuts = len(gt_cuts)
    gt_hard_cuts = Counter(map(itemgetter(2), gt_cuts))[1]
    gt_gradual_cuts = Counter(map(itemgetter(2), gt_cuts))[2]

    correct_preds = 0
    gradual_correct = 0
    hard_correct = 0

    gt_cuts_bool = [True] * len(gt_cuts)

    for begin1, end1, pred in pred_cuts:
        for idx, ground_truth in enumerate(gt_cuts):
            gt_begin2, gt_end2, label = ground_truth
            if check_overlap(begin1, end1, gt_begin2, gt_end2) and pred == label and gt_cuts_bool[idx]:
                correct_preds += 1
                if label == 1:
                    hard_correct += 1
                elif label == 2:
                    gradual_correct += 1
                gt_cuts_bool[idx] = False
                break

    prec = precision(correct_preds, num_preds)
    rec = recall(correct_preds, total_gt_cuts)
    f1 = f1_score(prec, rec)

    gradual_prec = precision(gradual_correct, gradual_preds)
    gradual_rec = recall(gradual_correct, gt_gradual_cuts)
    gradual_f1 = f1_score(gradual_prec, gradual_rec)

    hard_prec = precision(hard_correct, hard_preds)
    hard_rec = recall(hard_correct, gt_hard_cuts)
    hard_f1 = f1_score(hard_prec, hard_rec)

    print("Stats")
    print(f"Precision: {prec}, Recall: {rec}, F1-score: {f1}")
    print(f"Correct Predictions: {correct_preds} Total Predictions: {num_preds} GT Cuts: {total_gt_cuts}")
    print(f"Hard Cuts: Precision: {hard_prec}, Recall: {hard_rec}, F1-score: {hard_f1}")
    print(f"Hard Correct Predictions: {hard_correct} Hard Predictions: {hard_preds} Hard GT Cuts: {gt_hard_cuts}")
    print(f"Gradual Cuts: Precision: {gradual_prec}, Recall: {gradual_rec}, F1-score: {gradual_f1}")
    print(f"Gradual Correct Predictions: {gradual_correct} Gradual Predictions: {gradual_preds} Gradual GT Cuts: {gt_gradual_cuts}")

    print("\n")

    return prec, rec, f1


def test_videos_segment(videos_path, output_path, num_frames=16, overlap=8):
    videos = [x for x in os.listdir(videos_path) if x.endswith(".mp4")]
    videos = ["21829.mp4"]
    df = pd.DataFrame(columns=["Idx", "Film", "Type of Cut"])
    if not os.path.exists(os.path.join(output_path, "annotations.csv")):
        df.to_csv(os.path.join(output_path, "annotations.csv"), index=False)

    constant = 1000
    idx = 0
    for v in videos:
        frames = utils.video.get_frames(os.path.join(videos_path, v))
        annotations = gt_boundaries(os.path.join(videos_path, v.split(".mp4")[0] + "_gt.txt"))
        segments_iterator = segments_generator(frames, num_frames, overlap)
        print(annotations)

        for i, segment in enumerate(segments_iterator):
            label = 0
            type_of_cut = "No Transition"
            print(i * overlap, i * num_frames + num_frames)
            for transition in annotations:
                if check_overlap(transition[0], transition[1], i * overlap, i * overlap + num_frames):
                    label = transition[2]
                    if label == 1:
                        type_of_cut = "Hard"
                    else:
                        type_of_cut = "Gradual"
                    break
                elif (i * overlap + num_frames) < transition[0]:
                    break

            print(label)
            new_row = [idx, v.split(".mp4")[0], type_of_cut]
            df.loc[len(df)] = new_row
            utils.video.array_to_video(segment, 15, os.path.join(output_path, type_of_cut, str(idx) + ".mp4"))
            idx += 1

    df.to_csv(os.path.join(output_path, "annotations.csv"), mode="a", index=False, header=False)


# test_videos_segment("../../Data/RAI/ShotDetector/")


if __name__ == "__main__":
    # test_videos_segment("../../Data/RAI/ShotDetector/video_rai/", "../../Data/RAI/segments")

    model = deep_SBD.Model()
    model.load_state_dict(torch.load("../data/syn_data/model_final.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    inference.run(model, "../../Data/RAI/ShotDetector/video_rai/21829.mp4", "../../Results/21829.txt", 16, 8)
    evaluate_predictions("../../Results/21829.txt", "../../Data/RAI/ShotDetector/video_rai/21829_gt.txt")
