import argparse
from evaluation.inference import run
from models import deep_SBD
import torch
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='Required video path argument.')
    parser.add_argument('output_path', type=str, help='Path to save the results json file.')

    args = parser.parse_args()

    print("Model loading...")
    model = deep_SBD.Model()
    model.load_state_dict(torch.load("../checkpoints/deepSBD_newww10.pt")['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print("Inference device:", device)
    preds = run(model, 0.9, args.video_path, args.output_path, num_frames=16, overlap=8, width=112, height=112)

    pred_hard = [(begin, end) for begin, end, label in preds if label == 1]
    pred_graduals = [(begin, end) for begin, end, label in preds if label == 2]

    results = {"Hard": pred_hard, "Gradual": pred_graduals}

    with open(os.path.join(args.output_path, 'results.json'), 'w') as fp:
        json.dump(results, fp)
