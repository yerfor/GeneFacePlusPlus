import os

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import binary_erosion
from scipy.signal import medfilt

from utils.audio.pitch.extractor_utils import get_med_curve, clean_short_v_frag


def crepe_predict(audio, sr, model_capacity='full', center=True, step_size=10, verbose=1):
    from crepe.core import to_viterbi_cents, to_local_average_cents
    from crepe import get_activation
    np.seterr(divide='ignore', invalid='ignore')
    activation = get_activation(audio, sr, model_capacity=model_capacity,
                                center=center, step_size=step_size,
                                verbose=verbose)
    confidence = activation.max(axis=1)

    cents_v = to_viterbi_cents(activation)
    frequency_v = 10 * 2 ** (cents_v / 1200)
    frequency_v[np.isnan(frequency_v)] = 0

    cents = to_local_average_cents(activation)
    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency_v, frequency, confidence, activation


def load_model(device, capacity='full'):
    import torchcrepe
    # Bind model and capacity
    capacity = capacity
    model = torchcrepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(torchcrepe.__file__), 'assets', f'{capacity}.pth')
    model.load_state_dict(torch.load(file, map_location='cpu'))

    # Place on device
    model = model.to(torch.device(device))

    # Eval mode
    model.eval()
    return model


def crepe_predict_torch(audio, sr, hop_length=None, model_capacity='full',
                        batch_size=None, device='cpu', pad=True):
    from torchcrepe import preprocess, PITCH_BINS
    import warnings
    from crepe.core import to_viterbi_cents, to_local_average_cents

    warnings.filterwarnings('ignore', message=r'Named tensors and all their associated APIs.*')

    # Postprocessing breaks gradients, so just don't compute them
    with torch.no_grad():
        # Preprocess audio
        generator = preprocess(audio,
                               sr,
                               hop_length,
                               batch_size,
                               device,
                               pad)
        frames = next(generator)
        # Infer independent probabilities for each pitch bin
        model = load_model(device, model_capacity)
        model = model.to(frames.device)
        activation = model(frames)
        del model
        del frames

    # shape=(batch, 360, time / hop_length)
    activation = activation.reshape(-1, PITCH_BINS).cpu().numpy()
    torch.cuda.empty_cache()
    confidence = activation.max(axis=1)

    cents_v = to_viterbi_cents(activation)
    frequency_v = 10 * 2 ** (cents_v / 1200)
    frequency_v[np.isnan(frequency_v)] = 0

    cents = to_local_average_cents(activation)
    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    return frequency_v, frequency, confidence, activation


def cents_to_bins(cents):
    """Converts cents to pitch bins"""
    CENTS_PER_BIN = 20  # cents
    bins = (cents - 1997.3794084376191) / CENTS_PER_BIN
    return np.round(bins).astype(int)


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency))


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * np.log2(frequency / 10. + 1e-8)


def find_nearest_f0_in_piptrack(f0, pitches):
    i_frame = np.arange(len(f0))
    return pitches[i_frame, np.abs(f0[:, None] - pitches).argmin(-1)]


def f0_energy_corrector(wav_data_16k, f0_func, f0_min, f0_max, fix_octave_error=True):
    hop_size = 256
    win_size = hop_size * 6
    sr = 16000

    spec = np.abs(librosa.stft(wav_data_16k, n_fft=win_size, hop_length=hop_size,
                               win_length=win_size, pad_mode="constant").T)
    T = spec.shape[0]
    x_h256 = np.arange(0, 1, 1 / T)[:T]
    x_h256[-1] = 1
    f0 = f0_func(x_h256)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=win_size)
    x_idx = np.arange(T)

    def find_nearest_stft_bin(f0_):
        return np.abs(freqs[None, :] - f0_[:, None]).argmin(-1)

    def get_energy_mask(f0_lambda, hars=None, win_size=3):
        if hars is None:
            hars = [1]
        mask = np.zeros([T, 10000]).astype(bool)
        mask_bins = []
        for multiple in hars:
            f0_bin_idx = find_nearest_stft_bin(f0_lambda(f0, multiple))
            for delta in range(-win_size // 2, 1 + win_size // 2):
                y_idx = f0_bin_idx + delta
                if np.max(y_idx) < spec.shape[1]:
                    mask_bins.append(spec[x_idx, y_idx])
                mask[x_idx, y_idx] = 1
        mask_bins = np.stack(mask_bins, 1)
        energy_ = np.mean(mask_bins, 1)
        return energy_, mask

    bottom_idx = find_nearest_stft_bin(np.array([70]))[0]
    bottom_energy = spec[:, :bottom_idx].mean()
    pitches, _ = librosa.piptrack(
        wav_data_16k, sr,
        n_fft=win_size, win_length=win_size, hop_length=hop_size,
        fmin=50, fmax=3000, ref=bottom_energy)
    pitches = pitches.T[:T]
    f0_piptrack = find_nearest_f0_in_piptrack(f0, pitches)
    f0_raw = f0
    f0 = f0_piptrack

    # find uv first (for obtaining mean_energy_mharfhar)
    energy_har, mask_har = get_energy_mask(lambda f0, m: f0 * m, [1, 2], 3)
    energy_mhalfhar, mask_mhalfhar = get_energy_mask(lambda f0, m: f0 * (m - 0.5), [1], 5)
    r_energy = energy_har / np.clip(energy_mhalfhar, 1e-8, None)

    uv = np.zeros_like(f0).astype(bool)
    uv |= r_energy < 10
    uv |= (f0 > f0_max) | (f0 < f0_min)
    uv |= energy_har < bottom_energy
    mean_energy_mharfhar = np.clip(energy_mhalfhar[~uv].mean(), 1e-8, None)
    if len(uv) > 0:
        spec = np.clip(spec - spec[uv].mean(0)[None, :], 1e-8, None)

    # fix octave error
    r_energy_div_dict = {}
    if fix_octave_error:
        for div, mul, thres in [
            (2, (1,), 20),
            (3, (1, 2), 20),
            (5, (1, 2, 3), 20),
        ]:
            energy_div_har, mask_div_har = get_energy_mask(lambda f0, m: f0 / div * m, mul, 3)
            r_energy_div = energy_div_har / mean_energy_mharfhar
            r_energy_div = medfilt(r_energy_div, 5)

            r_energy_div_dict[div] = r_energy_div
            div_mask = (r_energy_div > thres) & (f0 / div > f0_min)
            f0[div_mask] /= div

            div_mask_erosion = binary_erosion(div_mask, iterations=2)
            div_pos = sorted(np.where(div_mask_erosion)[0])
            for pos in div_pos:
                for s in range(10):
                    if pos - s not in div_pos and pos - s >= 0:
                        f0[pos - s] = pitches[pos - s, np.abs(f0[pos] - pitches[pos - s]).argmin()]
                    if pos + s not in div_pos and pos + s < T:
                        f0[pos + s] = pitches[pos + s, np.abs(f0[pos] - pitches[pos + s]).argmin()]

    # find uv second
    energy_har, mask_har = get_energy_mask(lambda f0, m: f0 * m, [1, 2], 3)
    energy_mhalfhar, mask_mhalfhar = get_energy_mask(lambda f0, m: f0 * (m - 0.5), [1], 5)
    energy_har_2, _ = get_energy_mask(lambda f0, m: f0 * m, [2], 3)
    energy_mhalfhar_2, _ = get_energy_mask(lambda f0, m: f0 * (m - 0.5), [2, 3], 3)

    r_energy = energy_har / np.clip(energy_mhalfhar, 1e-8, None)
    r_energy = medfilt(r_energy, 3)
    r_energy_2 = energy_har_2 / np.clip(energy_mhalfhar_2, 1e-8, None)
    r_energy_2 = medfilt(r_energy_2, 3)
    r_energy_2_mask = r_energy_2 < 3
    r_energy_2_mask = binary_erosion(r_energy_2_mask, iterations=3)

    uv = np.zeros_like(f0).astype(bool)
    uv |= r_energy < 8
    uv |= r_energy_2_mask
    uv |= (f0 > f0_max) | (f0 < f0_min)
    uv |= energy_har < bottom_energy

    func_uv = interp1d(x_h256, uv, 'nearest')
    func_f0_div = interp1d(x_h256, f0, 'nearest')

    spec_log = np.log10(spec + 1e-8)

    return func_uv, func_f0_div, {
        'spec': spec_log,
        'energy_har': energy_har, 'energy_halfhar': energy_mhalfhar,
        'r_energy': r_energy, 'r_energy_2': r_energy_2,
        'mask_har': mask_har, 'mask_halfhar': mask_mhalfhar,
        'bottom_energy': bottom_energy,
        'r_energy_div_dict': r_energy_div_dict,
        'f0_piptrack': f0_piptrack,
        'f0_raw': f0_raw
    }


def crepe_with_corrector(wav_data, hop_size, audio_sample_rate, f0_min, f0_max, return_states=False, *args, **kwargs):
    wav_data = wav_data.astype(np.double)
    wav_data_16k = librosa.resample(wav_data, audio_sample_rate, 16000)
    time, f0_10ms, f0_nov, confi, activation = crepe_predict(
        wav_data_16k, 16000, step_size=10, model_capacity='small', center=True, verbose=0)
    T_10ms = len(f0_10ms)
    x_10ms = np.arange(0, 1, 1 / T_10ms)[:T_10ms]
    x_10ms[-1] = 1.0
    func_f0 = interp1d(x_10ms, f0_10ms, 'nearest')

    n_mel_frames = int(len(wav_data) // hop_size)
    x_new = np.arange(0, 1, 1 / n_mel_frames)[:n_mel_frames]
    x_new[-1] = 1.0

    # correct f0 using energy spec (first round)
    func_uv, func_f0, states = f0_energy_corrector(wav_data_16k, func_f0, f0_min, f0_max, fix_octave_error=True)
    f0_10ms = func_f0(x_10ms)
    uv_10ms = (func_uv(x_10ms) > 1e-4) & (confi < 0.9)
    uv_10ms = medfilt(uv_10ms.astype(float), 3) > 1e-4
    states['activation'] = activation
    states['confidence'] = confi

    # viterbi by voiced chunk, to fix incorrect viterbi smoothing in UV border.
    f0_10ms[uv_10ms] = 0
    f0_10ms_new = np.zeros_like(f0_10ms).astype(float)
    v_begin = -1
    for i in range(T_10ms):
        if not uv_10ms[i] and i < T_10ms - 1:
            if v_begin == -1:
                v_begin = i
        elif v_begin != -1:
            v_end = i - 1 if uv_10ms[i] else i
            if v_end - v_begin > 3:
                f0_bins = frequency_to_bins(f0_10ms[v_begin:v_end + 1])
                for j, k in zip(range(v_begin, v_end + 1), f0_bins):
                    if f0_10ms[j] > 1e-4:
                        activation[j, k + 10:] /= 5
                cents_v = to_viterbi_cents(activation[v_begin:v_end + 1])
                f0__ = 10 * 2 ** (cents_v / 1200)
                f0__[np.isnan(f0__)] = 0
                f0_10ms_new[v_begin:v_end + 1] = f0__
                v_begin = -1
    f0_10ms = f0_10ms_new

    # remove pitch deviated from median
    f0_10ms[confi < 0.1] = 0
    try:
        x_med_curve, y_med_curve = get_med_curve(f0_10ms)
        f0_med_curve = interp1d(np.array(x_med_curve), np.array(y_med_curve), 'nearest')(np.arange(len(f0_10ms)))
        f0_10ms[(f0_10ms < f0_med_curve - 100) | (f0_10ms > f0_med_curve + 100)] = 0
        states['f0_med_curve'] = interp1d(x_10ms, f0_med_curve)(x_new)
    except:
        pass
        # print("| WARN: catch an Error in get_med_curve.")
        # traceback.print_exc()

    # correct f0 using energy spec (second round), for better UV
    func_f0 = interp1d(x_10ms, f0_10ms, 'nearest')
    func_uv, func_f0, states_ = f0_energy_corrector(wav_data_16k, func_f0, f0_min, f0_max, fix_octave_error=False)
    del states_['r_energy_div_dict']
    states.update(states_)

    # interpolate f0
    confi_new = interp1d(x_10ms, confi)(x_new)
    f0 = func_f0(x_new)
    uv = (clean_short_v_frag(f0) | (func_uv(x_new) > 1e-4)) & (confi_new < 0.9)
    uv = medfilt(uv.astype(float), 3) > 1e-4
    f0 = medfilt(f0, 3)
    f0[uv] = 0

    if return_states:
        return f0, states
    else:
        return f0
