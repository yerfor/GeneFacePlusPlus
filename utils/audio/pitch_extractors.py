import os
import subprocess
import tempfile
import traceback
import uuid

import torch
from scipy.signal import medfilt

from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import librosa
import numpy as np
from scipy.interpolate import interp1d
from utils.audio.pitch.crepe_utils import crepe_with_corrector, crepe_predict, frequency_to_bins, \
    crepe_predict_torch
from utils.audio.pitch.extractor_utils import find_nearest_stft_bin, find_best_f0_using_har_energy

PITCH_EXTRACTOR = {}


def register_pitch_extractor(name):
    def register_pitch_extractor_(cls):
        PITCH_EXTRACTOR[name] = cls
        return cls

    return register_pitch_extractor_


def get_pitch_extractor(name):
    return PITCH_EXTRACTOR[name]


def extract_pitch_simple(wav):
    from utils.commons.hparams import hparams
    n_mel_frames = (len(wav) + 1) // hparams['hop_size'] - hparams['win_size'] // hparams['hop_size']
    return extract_pitch(hparams['pitch_extractor'], wav,
                         hparams['hop_size'], hparams['audio_sample_rate'],
                         f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
                         n_mel_frames=n_mel_frames)


def extract_pitch(extractor_name, wav_data, hop_size, audio_sample_rate, f0_min=75, f0_max=800, **kwargs):
    return get_pitch_extractor(extractor_name)(wav_data, hop_size, audio_sample_rate, f0_min, f0_max, **kwargs)


@register_pitch_extractor('harvest')
def harvest(wav_data, hop_size, audio_sample_rate, *args, **kwargs):
    import pyworld as pw
    n_mel_frames = int(len(wav_data) // hop_size)
    f0, t = pw.harvest(wav_data.astype(np.double), audio_sample_rate)
    x_old = np.arange(0, 1, 1 / len(f0))[:len(f0)]
    x_old[-1] = 1.0
    x_new = np.arange(0, 1, 1 / n_mel_frames)[:n_mel_frames]
    f0 = interp1d(x_old, f0, 'nearest')(x_new)
    return f0


@register_pitch_extractor('dio')
def dio(wav_data, hop_size, audio_sample_rate, *args, **kwargs):
    import pyworld as pw
    n_mel_frames = int(len(wav_data) // hop_size)
    _f0, t = pw.dio(wav_data.astype(np.double), audio_sample_rate)
    f0 = pw.stonemask(wav_data.astype(np.double), _f0, t, audio_sample_rate)
    x_old = np.arange(0, 1, 1 / len(f0))[:len(f0)]
    x_old[-1] = 1.0
    x_new = np.arange(0, 1, 1 / n_mel_frames)[:n_mel_frames]
    f0 = interp1d(x_old, f0, 'nearest')(x_new)
    return f0


@register_pitch_extractor('parselmouth')
def parselmouth_pitch(wav_data, hop_size, audio_sample_rate, f0_min, f0_max,
                      voicing_threshold=0.45, *args, **kwargs):
    import parselmouth
    time_step = hop_size / audio_sample_rate * 1000
    n_mel_frames = int(len(wav_data) // hop_size)
    f0_pm = parselmouth.Sound(wav_data, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=voicing_threshold,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode='constant')
    return f0


@register_pitch_extractor('reaper')
def reaper_extract_f0(wav_data, hop_size, audio_sample_rate, f0_min, f0_max, denoise=True,
                      return_denoised_wav=False,
                      *args, **kwargs):
    dirname = f'/tmp/reaper_tmp/{len(wav_data)}_{str(uuid.uuid1())}'
    os.makedirs(dirname, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dirname) as _:
        if hop_size == 256:
            if audio_sample_rate == 24000:
                save_wav(wav_data, f'{dirname}/1.wav', 25600, norm=False)
            if audio_sample_rate == 48000:
                save_wav(wav_data, f'{dirname}/1.wav', 51200, norm=False)
        else:
            assert hop_size == 240
            save_wav(wav_data, f'{dirname}/1.wav', audio_sample_rate, norm=False)

        if denoise:
            from utils.audio import trim_long_silences
            wav_data_ = wav_data
            _, audio_mask, sr = trim_long_silences(wav_data, audio_sample_rate, vad_max_silence_length=20)
            sr = audio_sample_rate
            wav_noise = wav_data[~audio_mask]
            # wav_noise = wav_data[:round(audio_sample_rate * 0.1)]

            # from scipy.signal import butter, lfilter
            # Define the filter parameters
            # cutoff_freq = 200.0  # Hz
            # nyquist_freq = 0.5 * sr
            # order = 5
            # b, a = butter(order, cutoff_freq / nyquist_freq, btype='lowpass')

            new_fn = f'{dirname}/0'
            save_wav(wav_noise, f'{new_fn}-noise.wav', sr=sr)
            save_wav(wav_data, f'{new_fn}.wav', sr=sr)
            subprocess.check_call(
                f'sox {new_fn}-noise.wav -n noiseprof {new_fn}-noise.prof; '
                f'sox {new_fn}.wav {new_fn}.denoised.wav noisered {new_fn}-noise.prof 0.21; ', shell=True)
            wav_data, _ = librosa.load(f'{new_fn}.denoised.wav', sr=sr)
            wav_data = np.concatenate([wav_data, wav_data_[-1024:]], 0)
            # wav_data = lfilter(b, a, wav_data)

        if hop_size == 256:
            if audio_sample_rate == 24000:
                save_wav(wav_data, f'{dirname}/2.wav', 25600, norm=False)
            if audio_sample_rate == 48000:
                save_wav(wav_data, f'{dirname}/2.wav', 51200, norm=False)
        else:
            assert hop_size == 240
            save_wav(wav_data, f'{dirname}/2.wav', audio_sample_rate, norm=False)

        retry = 10
        while retry > 0:
            subprocess.check_call(f'rm -rf {dirname}/*f0', shell=True)
            try:
                f0 = reaper_extract_f0_(
                    f'{dirname}/2.wav', f'{dirname}/1.wav', dirname, f0_min, f0_max)[:-8]
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                traceback.print_exc()
                retry -= 1
    if audio_sample_rate == 24000:
        if hop_size == 256:
            f0 = f0 * audio_sample_rate / 25600
        f0[f0 == 0] = -100000
        f0 = f0.reshape(-1, 2).mean(-1)
        f0[f0 < 0] = 0
    if audio_sample_rate == 48000:
        if hop_size == 256:
            f0 = f0 * audio_sample_rate / 51200
    if return_denoised_wav:
        return f0, wav_data
    else:
        return f0


def reaper_extract_f0_(fwav1, fwav2, temp_dir, pitch_lower, pitch_upper):
    frame_shift = 5
    use_reaper = True
    straight_f0_file = f'{temp_dir}/1.sf0'
    if not os.path.exists(straight_f0_file):
        subprocess.check_call('utils/audio/pitch/bin/ExtractF0ByStraight frame_shift=%d ' \
                              'min_f0=%d max_f0=%d wave="%s" output="%s"' % (
                                  frame_shift, pitch_lower, pitch_upper,
                                  fwav1, straight_f0_file), shell=True, timeout=20)
    if use_reaper:
        reaper_f0_file = f'{temp_dir}/1.rf0'
        if not os.path.exists(reaper_f0_file):
            subprocess.check_call('utils/audio/pitch/bin/ReaperF0 wave="%s" output="%s" ' \
                                  'f0_min=%d f0_max=%d' % (
                                      fwav2, reaper_f0_file,
                                      pitch_lower, pitch_upper), shell=True, timeout=20)  # ignore_security_alert
        interp_f0_file = f'{temp_dir}/1.tf0'
        if not os.path.exists(interp_f0_file):
            subprocess.check_call('utils/audio/pitch/bin/InterpF0 straight="%s" ' \
                                  'reaper="%s" output="%s"' % (
                                      straight_f0_file, reaper_f0_file, interp_f0_file), shell=True, timeout=20)
        straight_f0_file = interp_f0_file

    f0 = np.loadtxt(straight_f0_file, dtype=np.float32)
    return f0
