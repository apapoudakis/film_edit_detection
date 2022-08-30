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

    for v in videos:
        video_name = v.split(".")[0]
        print("Test Video:", video_name)

        # prediction
        run(model, threshold, test_video=os.path.join(videos_path, v), output_path=os.path.join("../../Results/",
            test_set, video_name + ".txt"), num_frames=16, overlap=8, width=width, height=height)

        # evaluation
        prec, rec, f1 = evaluate_predictions(os.path.join("../../Results/", test_set, video_name + ".txt"),
                                             os.path.join(videos_path, video_name + "_gt.txt"))

        # stats
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)

    print(f"Average Metrics")
    print(f"Precision: {sum(precision)/len(precision)} Recall: {sum(recall)/len(recall)} "
          f"F1-score: {sum(f1_score)/ len(f1_score)}")


if __name__ == "__main__":
    model = deep_SBD.Model()
    model.load_state_dict(torch.load("../../checkpoints/deepSBD_svm_checkpoint12.pt")['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    evaluate(model, test_set="RAI", threshold=0.9, width=112, height=112)
