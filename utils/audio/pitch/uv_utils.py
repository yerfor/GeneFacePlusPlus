import librosa
import numpy as np
from scipy.interpolate import interp1d


def uv_energy_corrector(wav_data_16k, f0_func, f0_min=50, f0_max=1000):
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

    # find uv first (for obtaining mean_energy_mharfhar)
    energy_har, mask_har = get_energy_mask(lambda f0, m: f0 * m, [1, 2], 3)
    energy_mhalfhar, mask_mhalfhar = get_energy_mask(lambda f0, m: f0 * (m - 0.5), [1], 5)
    r_energy = energy_har / np.clip(energy_mhalfhar, 1e-8, None)

    uv = np.zeros_like(f0).astype(bool)
    uv |= r_energy < 10
    uv |= (f0 > f0_max) | (f0 < f0_min)
    func_uv = interp1d(x_h256, uv, 'nearest', fill_value='extrapolate')
    return func_uv
