"""
Makes a FAKE Muse 2 dataset and saves it as CSV files in data/,
in the same layout a real Muse recording uses:

    timestamp, TP9, AF7, AF8, TP10      (256 rows per second, microvolts)

Files made:
    data/relax.csv          2 min, eyes closed, relaxed     (calibration low)
    data/task.csv           2 min, word-association task    (calibration high)
    data/lecture.csv        5 min lecture with 3 zone-outs
    data/lecture_truth.json when the fake student REALLY zoned out (for testing)

Run once:  python3 make_dataset.py
"""
import json
import os

import numpy as np
from scipy import signal

FS = 256
rng = np.random.default_rng(42)


def band_noise(n, low, high):
    """Random wiggles that only contain frequencies between low and high Hz.
    Real brain rhythms look like this: wobbly, not perfect sine waves."""
    b, a = signal.butter(4, [low, high], btype="bandpass", fs=FS)
    x = signal.filtfilt(b, a, rng.normal(size=n))
    return x / x.std()                       # scale to size 1


def pink_noise(n):
    """Background brain noise: stronger at low frequencies (1/f), like real EEG."""
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n, 1 / FS)
    spectrum[1:] /= np.sqrt(freqs[1:])
    x = np.fft.irfft(spectrum, n)
    return x / x.std()


def one_channel(focus, frontal):
    """focus: one value per sample, 0 = zoned out, 1 = locked in."""
    n = len(focus)
    theta = (6 - 3 * focus) * band_noise(n, 4, 8)      # drowsy / drifting
    alpha = (8 - 4 * focus) * band_noise(n, 8, 13)     # idle / zoned out
    beta = (2 + 3 * focus) * band_noise(n, 13, 30)     # thinking hard
    if not frontal:
        alpha *= 1.5        # ear sensors (TP9/TP10) pick up more alpha
    background = 5 * pink_noise(n)
    hum = 3 * np.sin(2 * np.pi * 60 * np.arange(n) / FS)   # power-line noise
    return 800 + theta + alpha + beta + background + hum  # Muse raw values sit around 800


def add_blinks(x, seconds):
    """Eye blinks: big slow bumps on the forehead sensors, every ~4 s."""
    for t in rng.uniform(0, seconds, size=int(seconds / 4)):
        i = int(t * FS)
        bump = 60 * np.hanning(int(0.3 * FS))
        x[i:i + len(bump)] += bump[: len(x) - i]
    return x


def make_recording(focus_per_second):
    focus = np.repeat(np.asarray(focus_per_second, float), FS)
    seconds = len(focus_per_second)
    channels = {}
    for name in ["TP9", "AF7", "AF8", "TP10"]:
        frontal = name.startswith("AF")
        x = one_channel(focus, frontal)
        channels[name] = add_blinks(x, seconds) if frontal else x
    channels["timestamp"] = np.arange(len(focus)) / FS
    return channels


def save_csv(path, ch):
    cols = ["timestamp", "TP9", "AF7", "AF8", "TP10"]
    data = np.column_stack([ch[c] for c in cols])
    np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="", fmt="%.4f")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    save_csv("data/relax.csv", make_recording([0.0] * 120))
    save_csv("data/task.csv", make_recording([1.0] * 120))

    # the fake student's "script" for the lecture
    zone_outs = [
        {"start": 40, "end": 80, "focus": 0.0},     # deep zone-out
        {"start": 130, "end": 165, "focus": 0.5},   # medium
        {"start": 210, "end": 245, "focus": 0.6},   # light
    ]
    focus = [1.0] * 300
    for z in zone_outs:
        focus[z["start"]:z["end"]] = [z["focus"]] * (z["end"] - z["start"])
    save_csv("data/lecture.csv", make_recording(focus))

    with open("data/lecture_truth.json", "w") as f:
        json.dump(zone_outs, f, indent=2)

    print("Saved data/relax.csv, data/task.csv, data/lecture.csv, data/lecture_truth.json")
