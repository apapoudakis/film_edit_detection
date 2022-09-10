import os
from inference import run
from helpers import evaluate_predictions
from models import deep_SBD2, deep_SBD
import torch


def evaluate(model, test_set, threshold, width, height):
    """
    Evaluate model on benchmark datasets (e.g. BBC, RAI)

    :param model:
    :param test_set:
    :param threshold:
    :param width:
    :param height:
    """

    if test_set == "RAI":
        videos_path = "../../Data/RAI/ShotDetector/video_rai/"
    elif test_set == "BBC":
        videos_path = "../../Data/BBC_Planet_Earth_Dataset/video/"
    else:
        print("Unknown test set!!")

    videos = [x for x in os.listdir(videos_path) if x.endswith(".mp4")]
    precision = []
    recall = []
    f1_score = []
    hard_precision = []
    hard_recall = []
    hard_f1_score = []
    gradual_precision = []
    gradual_recall = []
    gradual_f1_score = []

    for v in videos:
        video_name = v.split(".")[0]
        print("Test Video:", video_name)

        # prediction
        run(model, threshold, test_video=os.path.join(videos_path, v), output_path=os.path.join("../../Results/",
                                                                                                test_set,
                                                                                                video_name + ".txt"),
            num_frames=16, overlap=8, width=width, height=height, save=True)

        # evaluation
        overall_metrics, hard_metrics, gradual_metrics = evaluate_predictions(
            os.path.join("../../Results/", test_set, video_name + ".txt"),
            os.path.join(videos_path, video_name + "_gt.txt"))

        prec, rec, f1 = overall_metrics
        hard_prec, hard_rec, hard_f1 = hard_metrics
        gradual_prec, gradual_rec, gradual_f1 = gradual_metrics

        # stats
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)
        hard_precision.append(hard_prec)
        hard_recall.append(hard_rec)
        hard_f1_score.append(hard_f1)
        gradual_precision.append(gradual_prec)
        gradual_recall.append(gradual_rec)
        gradual_f1_score.append(gradual_f1)

    print(f"Average Metrics")
    print(f"Precision: {sum(precision) / len(precision)} Recall: {sum(recall) / len(recall)} "
          f"F1-score: {sum(f1_score) / len(f1_score)}")
    print(f"Hard Transitions")
    print(f"Precision: {sum(hard_precision) / len(hard_precision)} Recall: {sum(hard_recall) / len(hard_recall)} "
          f"F1-score: {sum(hard_f1_score) / len(hard_f1_score)}")
    print(f"Gradual Transitions")
    print(f"Precision: {sum(gradual_precision) / len(gradual_precision)} Recall: {sum(gradual_recall) / len(gradual_recall)} "
          f"F1-score: {sum(gradual_f1_score) / len(gradual_f1_score)}")


if __name__ == "__main__":
    model = deep_SBD.Model()
    model.load_state_dict(torch.load("../../checkpoints/deepSBD_new9.pt")['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    evaluate(model, test_set="RAI", threshold=0.9, width=112, height=112)
