import moviepy.editor as mp
import os
import torchaudio
import matplotlib.pyplot as plt
import torch


def extract_audio(video_file, save_path):
    video_name = os.path.basename(video_file).split(".mp4")[0]

    my_clip = mp.VideoFileClip(video_file)
    my_clip.audio.write_audiofile(os.path.join(save_path, video_name + ".wav"), )


def plot_audio(audio):

    waveform, sr = audio

    # check the number of channels
    if waveform.shape[0] == 1:
        plt.plot(waveform.t().numpy())
        plt.show()
    else:
        plt.figure(1)
        plt.plot(waveform[0, :].t().numpy())
        plt.figure(2)
        plt.plot(waveform[1, :].t().numpy())
    plt.show()


def save_spectogram(audio, output_path):

    waveform, sr = audio
    spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)
    print(spectrogram_tensor.shape)
    print(spectrogram_tensor.log2()[0, :, :].shape)
    plt.imsave("1.png", spectrogram_tensor.log2()[0, :, :].numpy(), cmap='viridis')
    plt.show()


def stereo_to_mono(audio):
    """
    Transform stereo to mono signals taking the average
    """
    waveform, sr = audio
    return torch.mean(waveform, dim=0, keepdim=True), sr


def resample(audio, new_sample_rate):
    waveform, sr = audio
    waveforms_list = []
    for channel in range(waveform.shape[0]):
        waveforms_list.append(torchaudio.transforms.Resample(sr, new_sample_rate)(waveform[channel, :]))

    waveform = torch.stack(waveforms_list)
    print(waveform.shape)
    return waveform, new_sample_rate
