"""
EEG kernel: Muse CSV -> theta/alpha/beta -> engagement score per second -> dips.
Follows the NeuroChat pipeline (Pope et al. engagement index E = beta / (alpha + theta)).
"""
import csv

import numpy as np
from scipy import signal

FS = 256                 # Muse 2 sampling rate (samples per second)
THETA = (4, 8)           # Hz
ALPHA = (8, 13)
BETA = (13, 30)


# ---------------------------------------------------------------- 0. load data
def load_muse_csv(path):
    """Reads a Muse CSV and returns one signal: the average of the two
    forehead sensors (AF7 + AF8), the best spots for attention.
    Works with our fake files AND Mind Monitor exports (columns like RAW_AF7)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]

    def find(name):
        for i, col in enumerate(header):
            if col.strip() in (name, f"RAW_{name}"):
                return i
        raise ValueError(f"No {name} column in {path}. Columns are: {header}")

    af7, af8 = find("AF7"), find("AF8")
    values = []
    for r in rows:
        try:
            values.append((float(r[af7]) + float(r[af8])) / 2)
        except (ValueError, IndexError):
            continue          # skip blank / broken rows (real files have some)
    return np.array(values)


# ---------------------------------------------------------------- 1. filter
def clean(eeg, fs=FS):
    """1-30 Hz bandpass + 60 Hz notch."""
    b, a = signal.butter(4, [1, 30], btype="bandpass", fs=fs)
    eeg = signal.filtfilt(b, a, eeg)
    b, a = signal.iirnotch(60, 30, fs=fs)
    return signal.filtfilt(b, a, eeg)


# ---------------------------------------------------------------- 2. epochs + FFT
def band_power(epoch, fs, band):
    freqs = np.fft.rfftfreq(len(epoch), 1 / fs)
    power = np.abs(np.fft.rfft(epoch * np.hanning(len(epoch)))) ** 2
    mask = (freqs >= band[0]) & (freqs < band[1])
    return power[mask].sum()


def band_powers(eeg, fs=FS, win=1.0, step=0.25):
    """Slide a 1 s window every 250 ms. For each window, measure how loud
    theta, alpha and beta are, and compute E = beta / (alpha + theta)."""
    n, hop = int(win * fs), int(step * fs)
    rows = []
    for start in range(0, len(eeg) - n + 1, hop):
        ep = eeg[start:start + n]
        th, al, be = (band_power(ep, fs, b) for b in (THETA, ALPHA, BETA))
        rows.append({"time": (start + n) / fs,            # end of the window
                     "theta": th, "alpha": al, "beta": be,
                     "E": be / (al + th)})
    return rows


# ---------------------------------------------------------------- 3. calibrate + normalize
def calibrate(E_relax, E_task):
    """5th/95th percentiles instead of raw min/max so one noisy window can't wreck the scale."""
    return np.percentile(E_relax, 5), np.percentile(E_task, 95)


def normalize(E, E_min, E_max):
    return np.clip((np.asarray(E) - E_min) / (E_max - E_min), 0, 1)


# ---------------------------------------------------------------- 4. smooth
def smooth(times, E_norm, window=15):
    """Every whole second, average the last `window` seconds of scores."""
    times = np.asarray(times)
    out = []
    for sec in range(1, int(times[-1]) + 1):
        recent = E_norm[(times > sec - window) & (times <= sec)]
        out.append(recent.mean())
    return np.array(out)                      # out[i] = score at second i+1


# ---------------------------------------------------------------- 5. dips
def find_dips(scores, threshold=0.4, min_len=15):
    """Windows where the per-second score stays below threshold for >= min_len seconds."""
    dips, start = [], None
    for sec, s in enumerate(list(scores) + [1.0]):     # sentinel closes a dip at the end
        if s < threshold and start is None:
            start = sec
        elif s >= threshold and start is not None:
            if sec - start >= min_len:
                dips.append({"start": start, "end": sec,
                             "score": round(float(np.mean(scores[start:sec])), 2)})
            start = None
    return dips


# ---------------------------------------------------------------- whole pipeline
def process_session(lecture_eeg, relax_eeg, task_eeg, fs=FS):
    """Returns (rows, scores, dips).
    rows   = theta/alpha/beta/E/E_norm for every 250 ms window
    scores = smoothed engagement, one per second
    dips   = moments attention stayed low"""
    E_relax = [r["E"] for r in band_powers(clean(relax_eeg, fs), fs)]
    E_task = [r["E"] for r in band_powers(clean(task_eeg, fs), fs)]
    E_min, E_max = calibrate(E_relax, E_task)

    rows = band_powers(clean(lecture_eeg, fs), fs)
    E_norm = normalize([r["E"] for r in rows], E_min, E_max)
    for r, en in zip(rows, E_norm):
        r["E_norm"] = float(en)

    scores = smooth([r["time"] for r in rows], E_norm)
    return rows, scores, find_dips(scores)


def save_scores(path, rows):
    """Save theta/alpha/beta/E/E_norm for every window so you can open it in Excel."""
    cols = ["time", "theta", "alpha", "beta", "E", "E_norm"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: round(r[c], 4) for c in cols})
