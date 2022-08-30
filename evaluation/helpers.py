"""
This repository https://github.com/Tangshitao/ClipShots/blob/master/tools/evaluate.py is used for
Shot Boundary Detection evaluation.
"""

import os
import re
from ast import literal_eval
from models import deep_SBD
from evaluation import inference
import utils
import pandas as pd
import torch
from utils.video import array_to_video


def check_overlap(begin1, end1, begin2, end2):
    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 >= begin2


def get_union_cnt(set1, set2):
    cnt = 0
    gt_cuts_bool = [True] * len(set2)
    for begin, end in set1:
        for idx, cut in enumerate(set2):
            _begin, _end = cut
            if check_overlap(begin, end, _begin, _end) and gt_cuts_bool[idx]:
                cnt += 1
                gt_cuts_bool[idx] = False
                break
    return cnt


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


def recall_pre_f1(a, b, c):
    recall = a / b if b != 0 else 0
    precison = a / c if c != 0 else 0
    f1 = 2 * recall * precison / (recall + precison)
    return precison, recall, f1


def evaluate_predictions(pred_file, gt_file):

    with open(pred_file) as f:
        pred_cuts = f.readlines()
    pred_cuts = [literal_eval(x.strip()) for x in pred_cuts]

    gt_cuts = gt_boundaries(gt_file)

    _gt_cuts = [(begin, end) for begin, end, _ in gt_cuts]
    gt_hard = [(begin, end) for begin, end, _ in gt_cuts if end - begin == 1]
    gt_graduals = [(begin, end) for begin, end, _ in gt_cuts if end - begin > 1]
    pred_hard = [(begin, end) for begin, end, label in pred_cuts if label == 1]
    pred_graduals = [(begin, end) for begin, end, label in pred_cuts if label == 2]

    hard_correct = get_union_cnt(gt_hard, pred_hard)
    gradual_correct = get_union_cnt(gt_graduals, pred_graduals)
    all_correct = get_union_cnt(_gt_cuts, pred_hard + pred_graduals)

    prec, rec, f1 = recall_pre_f1(all_correct, len(gt_hard) + len(gt_graduals), len(pred_hard) + len(pred_graduals))
    prec_hard, rec_hard, f1_hard = recall_pre_f1(hard_correct, len(gt_hard), len(pred_hard))
    prec_gradual, rec_gradual, f1_gradual = recall_pre_f1(gradual_correct, len(gt_graduals), len(pred_graduals))

    print(f"Precision: {prec}, Recall: {rec}, F1-score: {f1}")
    print(f"Correct Predictions: {all_correct} Total Predictions: {len(pred_hard) + len(pred_graduals)} GT Cuts: {len(gt_hard) + len(gt_graduals)}")
    print(f"Hard Cuts: Precision: {prec_hard}, Recall: {rec_hard}, F1-score: {f1_hard}")
    print(f"Hard Correct Predictions: {hard_correct} Hard Predictions: {len(pred_hard)} Hard GT Cuts: {len(gt_hard)}")
    print(f"Gradual Cuts: Precision: {prec_gradual}, Recall: {rec_gradual}, F1-score: {f1_gradual}")
    print(f"Gradual Correct Predictions: {gradual_correct} Gradual Predictions: {len(pred_graduals)} Gradual GT Cuts: {len(gt_graduals)}")

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
        segments_iterator = inference.segments_generator(frames, num_frames, overlap)
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
