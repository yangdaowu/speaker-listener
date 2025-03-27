import glob
import os
import os.path as osp
import argparse
from tqdm import tqdm
import cv2
import torch
import numpy as np
from multiprocessing import Pool
import soundfile as sf
import torchaudio
from scipy.io import loadmat
import torch.nn.functional as F

torchaudio.set_audio_backend("sox_io")

def extract_audio_features(audio_fn, recons_folder, output_folder):
    video_id = osp.basename(audio_fn)[:-4]
    fps = 30

    audio, sr = sf.read(audio_fn)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    n_frames = len(loadmat(f'{recons_folder}/test/{video_id}.speaker.mat')['coeff'])
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])

    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

    model = torchaudio.models.wav2vec2_base()
    model.eval()
    with torch.no_grad():
        features, _ = model.extract_features(audio_tensor)
        speech_features = features[-1]

    speech_features = F.interpolate(speech_features.transpose(1, 2), size=n_frames, mode='linear', align_corners=False)
    speech_features = speech_features.transpose(1, 2)

    projection = torch.nn.Linear(speech_features.size(-1), 256)
    projected_features = projection(speech_features)  # shape: (1, n_frames, 256)

    final_features = projected_features.squeeze(0).cpu().numpy()
    with open(f'{output_folder}/{video_id}.npy', 'wb') as f:
        np.save(f, final_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_folder', type=str, required=True)
    parser.add_argument('--input_recons_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    total_audio_fns = glob.glob(f'{args.input_audio_folder}/*.wav')

    for audio_fn in tqdm(total_audio_fns):
        extract_audio_features(audio_fn, args.input_recons_folder, args.output_folder)
