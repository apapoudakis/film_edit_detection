import os
import numpy as np
from PIL import Image
from numpy import asarray
from utils.video import array_to_video
import pandas as pd


def data_formatting(root_dir, annotation_path, width, height):

    data_segments = ["tv7789", "tv2001", "tv2007t", "tv2007d", "tv2008", "tv2009"]
    idx = 0

    df = pd.DataFrame(columns=["Idx", "Type of Cut"])
    if not os.path.exists(os.path.join(annotation_path, "annotations.csv")):
        df.to_csv(os.path.join(annotation_path, "annotations.csv"), index=False)
    for ds in data_segments:
        print(ds)
        for r, dirs, files in os.walk(os.path.join(root_dir, ds)):
            if files:
                frames = []
                files = [x for x in files if x.endswith(".jpg")]
                for f in sorted(files, key=lambda x: list(map(int, x.split(".")[0]))):
                    frames.append(asarray(Image.open(os.path.join(r, f)).resize((width, height))))
                video = np.stack(frames)
                folders = r.split("/")
                if "graduals" in folders:
                    label = "Gradual"
                elif "segments" in folders:
                    label = "No Transition"
                else:
                    label = "Hard"
                array_to_video(video, 10, os.path.join(annotation_path, label, str(idx) + ".mp4"))
                new_row = [str(idx), label]
                idx += 1
                df.loc[len(df)] = new_row

    df.to_csv(os.path.join(annotation_path, "annotations.csv"), mode="a", index=False, header=False)


data_formatting("../../Data/DeepSBD/", "../../Data/DeepSBD/EditedDeepSBD/", 112, 112)
