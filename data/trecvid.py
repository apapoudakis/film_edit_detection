import os
import xml.etree.ElementTree
import wget

dataset_path = "../../Data/TRECVID/"
sep_string = "._-o-_."


def download_video():
    """
    Download TRECVID IACC.3 videos
    """

    video_path = os.path.join(dataset_path, "Video")

    if not os.path.exists(video_path):
        os.mkdir(video_path)

    trecvid = xml.etree.ElementTree.parse(os.path.join(dataset_path, "iacc.3.collection.xml"))

    for video in trecvid.getroot():
        source = video.find("source").text
        fn = video.find("filename").text.split(sep_string)[1]
        print(fn)

        if not os.path.exists(os.path.join(video_path, fn)):
            try:
                wget.download(os.path.join(source, fn), out=os.path.join(video_path, fn))
            except:
                print("Error!!!")


if __name__ == "__main__":
    download_video()
