import librosa
import numpy as np
import pyloudnorm as pyln
import torch
from scipy.signal import get_window

from utils.audio.dct import dct
from utils.audio.vad import trim_long_silences


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


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return 10.0 ** (x * 0.05)


def normalize(S, min_level_db):
    return (S - min_level_db) / -min_level_db


def denormalize(D, min_level_db):
    return (D * -min_level_db) + min_level_db


def librosa_wav2spec(wav_path,
                     fft_size=None,
                     hop_size=256,
                     win_length=1024,
                     window="hann",
                     num_mels=80,
                     fmin=80,
                     fmax=-1,
                     eps=1e-6,
                     sample_rate=22050,
                     loud_norm=False,
                     trim_long_sil=False,
                     center=True):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path
    if fft_size is None:
        fft_size = win_length
    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -16.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=center)
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    # calculate mel spec
    mel = mel_basis @ linear_spc
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    if center:
        l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
        wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
        wav = wav[:mel.shape[1] * hop_size]

    # log linear spec
    linear_spc = np.log10(np.maximum(eps, linear_spc))
    return {'wav': wav, 'mel': mel.T, 'linear': linear_spc.T, 'mel_basis': mel_basis}


def librosa_wav2mfcc(wav_path,
                     fft_size=None,
                     hop_size=256,
                     win_length=1024,
                     window="hann",
                     num_mels=80,
                     fmin=80,
                     fmax=-1,
                     sample_rate=22050,
                     center=True):
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path
    mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13,
                                n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax,
                                hop_length=hop_size,
                                win_length=win_length, window=window, center=center)
    return mfcc.T


def torch_wav2spec(wav,
                   mel_basis,
                   fft_size=1024,
                   hop_size=256,
                   win_length=1024,
                   eps=1e-6):
    fft_window = get_window('hann', win_length, fftbins=True)
    fft_window = torch.FloatTensor(fft_window).to(wav.device)
    mel_basis = torch.FloatTensor(mel_basis).to(wav.device)
    x_stft = torch.stft(wav, fft_size, hop_size, win_length, fft_window,
                        center=False, pad_mode='constant', normalized=False, onesided=True, return_complex=True)
    linear_spc = torch.abs(x_stft)
    mel = mel_basis @ linear_spc
    mel = torch.log10(torch.clamp_min(mel, eps))  # (n_mel_bins, T)
    return mel.transpose(1, 2)


def mel2mfcc_torch(mel, n_coef=13):
    return dct(mel, norm='ortho')[:, :, :n_coef]


def librosa_wav2linearspec(wav_path,
                     fft_size=None,
                     hop_size=256,
                     win_length=1024,
                     window="hann",
                     num_mels=80,
                     fmin=80,
                     fmax=-1,
                     eps=1e-6,
                     sample_rate=22050,
                     loud_norm=False,
                     trim_long_sil=False,
                     center=True):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path
    if fft_size is None:
        fft_size = win_length
    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -16.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=center)
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # pad wav
    if center:
        l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
        wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
        wav = wav[:linear_spc.shape[1] * hop_size]

    # log linear spec
    linear_spc = np.log10(np.maximum(eps, linear_spc))
    return {'wav': wav, 'linear': linear_spc.T}


def librosa_linear2mel(linear_spec, hparams, num_mels=160, eps=1e-6):
    
    fft_size=hparams['fft_size']
    hop_size=hparams['hop_size']
    win_length=hparams['win_size']
    fmin=hparams['fmin']
    fmax=hparams['fmax']
    sample_rate=hparams['audio_sample_rate']

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel_basis = torch.FloatTensor(mel_basis).to(linear_spec.device)[None, :].repeat(linear_spec.shape[0], 1, 1)

    # perform linear spec to mel spec
    linear_spec = torch.pow(10, linear_spec)
    mel = torch.bmm(mel_basis, linear_spec.transpose(1, 2))
    mel = torch.log10(torch.clamp_min(mel, eps))  # (n_mel_bins, T)
    return mel.transpose(1, 2)

