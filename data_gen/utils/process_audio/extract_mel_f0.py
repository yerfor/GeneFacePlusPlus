import numpy as np
import torch
import glob
import os
import tqdm
import librosa
import parselmouth
from utils.commons.pitch_utils import f0_to_coarse
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.commons.os_utils import multiprocess_glob
from utils.audio.io import save_wav

from moviepy.editor import VideoFileClip
from utils.commons.hparams import hparams, set_hparams

def resample_wav(wav_name, out_name, sr=16000):
    wav_raw, sr = librosa.core.load(wav_name, sr=sr)
    save_wav(wav_raw, out_name, sr)
    
def split_wav(mp4_name, wav_name=None):
    if wav_name is None:
        wav_name = mp4_name.replace(".mp4", ".wav").replace("/video/", "/audio/")
    if os.path.exists(wav_name):
        return wav_name
    os.makedirs(os.path.dirname(wav_name), exist_ok=True)
    
    video = VideoFileClip(mp4_name,verbose=False)
    dur = video.duration
    audio = video.audio 
    assert audio is not None
    audio.write_audiofile(wav_name,fps=16000,verbose=False,logger=None)
    return wav_name

def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2

def extract_mel_from_fname(wav_path,
                      fft_size=512,
                      hop_size=320,
                      win_length=512,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-6,
                      sample_rate=16000,
                      min_level_db=-100):
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=False)
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = mel_basis @ spc

    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    mel = mel.T

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)

    return wav.T, mel

def extract_f0_from_wav_and_mel(wav, mel,
                        hop_size=320,
                        audio_sample_rate=16000,
                        ):
    time_step = hop_size / audio_sample_rate * 1000
    f0_min = 80
    f0_max = 750
    f0 = parselmouth.Sound(wav, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse


def extract_mel_f0_from_fname(wav_name=None, out_name=None):
    try:
        out_name = wav_name.replace(".wav", "_mel_f0.npy").replace("/audio/", "/mel_f0/")
        os.makedirs(os.path.dirname(out_name), exist_ok=True)

        wav, mel = extract_mel_from_fname(wav_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        out_dict = {
            "mel": mel, # [T, 80]
            "f0": f0,
        }
        np.save(out_name, out_dict)
    except Exception as e:
        print(e)

def extract_mel_f0_from_video_name(mp4_name, wav_name=None, out_name=None):
    if mp4_name.endswith(".mp4"):
        wav_name = split_wav(mp4_name, wav_name)
        if out_name is None:
            out_name = mp4_name.replace(".mp4", "_mel_f0.npy").replace("/video/", "/mel_f0/")
    elif mp4_name.endswith(".wav"):
        wav_name = mp4_name
        if out_name is None:
            out_name = mp4_name.replace(".wav", "_mel_f0.npy").replace("/audio/", "/mel_f0/")

    os.makedirs(os.path.dirname(out_name), exist_ok=True)

    wav, mel = extract_mel_from_fname(wav_name)

    f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
    out_dict = {
        "mel": mel, # [T, 80]
        "f0": f0,
    }
    np.save(out_name, out_dict)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--video_id', type=str, default='May', help='')
    args = parser.parse_args()
    ### Process Single Long Audio for NeRF dataset
    person_id = args.video_id

    wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
    out_name = f"data/processed/videos/{person_id}/aud_mel_f0.npy"
    extract_mel_f0_from_video_name(wav_16k_name, out_name)
    print(f"Saved at {out_name}")