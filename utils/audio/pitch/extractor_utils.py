import numba as nb
import numpy as np


@nb.njit()
def find_nearest_stft_bin(f0_, freqs):
    freqs = np.expand_dims(freqs, 0)
    f0_ = np.expand_dims(f0_, 1)
    return np.abs(freqs - f0_).argmin()


@nb.njit()
def get_med_curve(f0, step_size=20):
    v_begin = -1
    v_end = -1
    x_med_curve = []
    y_med_curve = []
    T = len(f0)

    for i in range(T):
        if f0[i] >= 50 and i < T - 1:
            if v_begin == -1:
                v_begin = i
            v_end = i
        else:
            if v_end != -1:
                if v_end - v_begin > 3:
                    for j in range(v_begin, v_end + 1 - step_size, step_size):
                        frag_med = np.median(f0[j:j + step_size])
                        x_med_curve.append(j)
                        y_med_curve.append(frag_med)
                    x_med_curve.append(v_end)
                    y_med_curve.append(np.median(f0[v_end - step_size:v_end + 1]))
            v_end = v_begin = -1
    x_med_curve = [0] + x_med_curve + [T]
    x_med_curve = np.array(x_med_curve)
    y_med_curve = [y_med_curve[0]] + y_med_curve + [y_med_curve[-1]]
    y_med_curve = np.array(y_med_curve)
    return x_med_curve, y_med_curve


@nb.njit()
def clean_short_v_frag(f0):
    v_begin = -1
    T = len(f0)

    uv = np.zeros_like(f0).astype(np.bool_)
    for i in range(T):
        if f0[i] >= 1e-4 and i < T - 1:
            if v_begin == -1:
                v_begin = i
        else:
            if v_begin != -1:
                v_end = i if f0[i] >= 1e-4 else i - 1
                if v_end - v_begin + 1 < 3:
                    uv[v_begin:v_end + 1] = 1
            v_begin = -1
    return uv


@nb.njit()
def find_best_f0_using_har_energy(spec, pitches, freqs, hars, hars_mhalf, f0_min, f0_max):
    re = np.zeros_like(spec)
    T = len(spec)
    for i in range(T):
        spec_i = spec[i]
        for j, f0_j in enumerate(pitches[i]):
            if f0_j == 0 or f0_j < f0_min[i] or f0_j > f0_max[i]:
                continue
            mask = np.zeros((10000,))
            mask_mhalf = np.zeros((10000,))
            for mul in hars:
                b = find_nearest_stft_bin(np.array((f0_j * mul,)), freqs)
                for delta in range(-1, 2):
                    mask[b + delta] = 1
            for mul in hars_mhalf:
                b_mhalf = find_nearest_stft_bin(np.array((f0_j * (mul - 0.5),)), freqs)
                for delta in range(-1, 2):
                    mask_mhalf[b_mhalf + delta] = 1
            mask = mask[:len(spec_i)]
            mask_mhalf = mask_mhalf[:len(spec_i)]
            energy = (np.exp(spec_i) * mask).sum() / mask.sum()
            energy_mhalf = (np.exp(spec_i) * mask_mhalf).sum() / mask_mhalf.sum()
            re[i, j] = energy / energy_mhalf
    f0_2d_mask = 10000 * (re > 2) + 20000 * (re > 3) + np.expand_dims(np.arange(re.shape[1])[::-1], 0)
    f0_idx = np.zeros((T,), dtype=np.int_)
    for i in range(T):
        f0_idx[i] = f0_2d_mask[i].argmax()
    uv = re.sum(-1) == 0

    f0 = np.zeros((T,))
    for i in range(T):
        f0[i] = pitches[i, f0_idx[i]]
    f0 = f0 * (1 - uv)
    uv = clean_short_v_frag(f0)
    f0[uv] = 0
    x_med_curve, y_med_curve = get_med_curve(f0)
    re = re * (re > 1.5)
    return re, f0, x_med_curve, y_med_curve
