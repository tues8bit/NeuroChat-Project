"""
EEG kernel: raw EEG -> engagement score per second -> dips.
Follows the NeuroChat pipeline (Pope et al. engagement index).
"""
import numpy as np
from scipy import signal
 
FS = 256                 # Muse 2 sampling rate (samples per second)
THETA = (4, 8)           # Hz
ALPHA = (8, 13)
BETA = (13, 30)
 
 
# ---------------------------------------------------------------- synthetic data
def make_fake_eeg(focus, fs=FS, seed=0):
    """focus: one value per second, 0 = zoned out, 1 = locked in.
    Returns 1 channel of fake EEG (microvolts) that behaves like the real thing:
    focused -> more beta, zoned out -> more alpha + theta.
    Also adds junk the filter must remove: 60 Hz power-line hum and slow drift."""
    rng = np.random.default_rng(seed)
    f = np.repeat(np.asarray(focus, dtype=float), fs)   # stretch to one value per sample
    t = np.arange(len(f)) / fs
 
    theta = (3 - 1.5 * f) * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6.28))
    alpha = (4 - 2 * f) * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28))
    beta = (1.5 + 1.5 * f) * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 6.28))
    noise = rng.normal(0, 2, len(t))
    hum = 5 * np.sin(2 * np.pi * 60 * t)                 # power-line noise
    drift = 20 * np.sin(2 * np.pi * 0.1 * t)             # sweat / movement drift
    return theta + alpha + beta + noise + hum + drift
 
 
# ---------------------------------------------------------------- 1. filter
def clean(eeg, fs=FS):
    """1-30 Hz bandpass + 60 Hz notch."""
    b, a = signal.butter(4, [1, 30], btype="bandpass", fs=fs)
    eeg = signal.filtfilt(b, a, eeg)
    b, a = signal.iirnotch(60, 30, fs=fs)
    return signal.filtfilt(b, a, eeg)
 
 
# ---------------------------------------------------------------- 2. epochs + FFT + E
def band_power(epoch, fs, band):
    freqs = np.fft.rfftfreq(len(epoch), 1 / fs)
    power = np.abs(np.fft.rfft(epoch * np.hanning(len(epoch)))) ** 2
    mask = (freqs >= band[0]) & (freqs < band[1])
    return power[mask].sum()
 
 
def engagement_index(eeg, fs=FS, win=1.0, step=0.25):
    """Slide a 1 s window every 250 ms. Returns (times, E) where E = beta / (alpha + theta)."""
    n, hop = int(win * fs), int(step * fs)
    times, E = [], []
    for start in range(0, len(eeg) - n + 1, hop):
        ep = eeg[start:start + n]
        th, al, be = (band_power(ep, fs, b) for b in (THETA, ALPHA, BETA))
        E.append(be / (al + th))
        times.append((start + n) / fs)       # time at the END of the window
    return np.array(times), np.array(E)
 
 
# ---------------------------------------------------------------- 3. calibrate + normalize
def calibrate(E_relax, E_task):
    """Use 5th/95th percentiles instead of raw min/max so one noisy window can't wreck the scale."""
    return np.percentile(E_relax, 5), np.percentile(E_task, 95)
 
 
def normalize(E, E_min, E_max):
    return np.clip((E - E_min) / (E_max - E_min), 0, 1)
 
 
# ---------------------------------------------------------------- 4. smooth
def smooth(times, E_norm, window=15):
    """Every whole second, average the last `window` seconds of scores."""
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
    _, E_relax = engagement_index(clean(relax_eeg, fs), fs)
    _, E_task = engagement_index(clean(task_eeg, fs), fs)
    E_min, E_max = calibrate(E_relax, E_task)
 
    times, E = engagement_index(clean(lecture_eeg, fs), fs)
    scores = smooth(times, normalize(E, E_min, E_max))
    return scores, find_dips(scores)
 
