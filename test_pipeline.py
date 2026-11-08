"""
Tests that don't need Ollama. Run with:  python3 -m pytest -v
Run  python3 make_dataset.py  first so the data/ files exist.
"""
import json

import numpy as np

from eeg_kernel import (FS, clean, band_powers, normalize, smooth, find_dips,
                        load_muse_csv, process_session)
from controller import get_tier, get_text, build_prompt, load_session


# ---------------- the data files

def test_csv_loads_with_right_length():
    lecture, relax, task = load_session()
    assert len(lecture) == 300 * FS        # 5 min at 256 samples/s
    assert len(relax) == 120 * FS
    assert len(task) == 120 * FS


def test_loader_reads_mind_monitor_column_names(tmp_path):
    f = tmp_path / "mm.csv"
    f.write_text("TimeStamp,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10\n0,1,10,20,4\n1,1,30,40,4\n,,,,\n")
    assert list(load_muse_csv(f)) == [15, 35]   # averages AF7+AF8, skips the blank row


# ---------------- alpha / beta / theta behave like a real brain

def band_means(eeg):
    rows = band_powers(clean(eeg))
    return {b: np.mean([r[b] for r in rows]) for b in ("theta", "alpha", "beta", "E")}


def test_relaxed_has_more_alpha_and_theta():
    _, relax, task = load_session()
    r, t = band_means(relax), band_means(task)
    assert r["alpha"] > t["alpha"]
    assert r["theta"] > t["theta"]


def test_task_has_more_beta():
    _, relax, task = load_session()
    assert band_means(task)["beta"] > band_means(relax)["beta"]


def test_engagement_higher_during_task():
    _, relax, task = load_session()
    assert band_means(task)["E"] > band_means(relax)["E"]


def test_filter_removes_60hz_hum():
    t = np.arange(10 * FS) / FS
    hum = np.sin(2 * np.pi * 60 * t)
    cleaned = clean(hum)[FS:-FS]                 # skip 1 s at each edge (filters wobble there)
    assert np.abs(cleaned).max() < 0.05


def test_windows_every_quarter_second():
    rows = band_powers(np.random.randn(10 * FS))
    assert len(rows) == 37                       # (10 s - 1 s window) / 0.25 s + 1
    assert np.allclose(np.diff([r["time"] for r in rows]), 0.25)


def test_normalize_stays_between_0_and_1():
    out = normalize([-5, 0.5, 99], 0, 1)
    assert out.min() >= 0 and out.max() <= 1


def test_smooth_gives_one_score_per_second():
    times = np.arange(1, 61, 0.25)
    assert np.allclose(smooth(times, np.ones_like(times)), np.ones(60))


# ---------------- dip detection edge cases

def test_dip_exactly_15_seconds_counts():
    assert len(find_dips([0.9] * 10 + [0.1] * 15 + [0.9] * 10)) == 1


def test_dip_14_seconds_does_not_count():
    assert find_dips([0.9] * 10 + [0.1] * 14 + [0.9] * 10) == []


def test_dip_at_very_end_is_caught():
    assert find_dips([0.9] * 10 + [0.1] * 20) == [{"start": 10, "end": 30, "score": 0.1}]


def test_no_dips_when_always_focused():
    assert find_dips([0.9] * 100) == []


# ---------------- full pipeline vs. the answer key

def test_finds_every_real_zone_out_and_nothing_else():
    _, _, dips = process_session(*load_session())
    truth = json.load(open("data/lecture_truth.json"))
    assert len(dips) == len(truth)
    for dip, real in zip(dips, truth):
        # detected dip overlaps the real one (smoothing makes it a few seconds late)
        assert dip["start"] < real["end"] and dip["end"] > real["start"]


def test_deeper_zone_out_gives_lower_score():
    _, _, dips = process_session(*load_session())
    scores = [d["score"] for d in dips]          # truth is ordered deep, medium, light
    assert scores == sorted(scores)


def test_deep_zone_out_is_very_low_tier():
    _, _, dips = process_session(*load_session())
    assert get_tier(dips[0]["score"])["name"] == "very low"


# ---------------- controller

def test_tier_boundaries():
    assert get_tier(0.0)["name"] == "very low"
    assert get_tier(0.199)["name"] == "very low"
    assert get_tier(0.2)["name"] == "low"
    assert get_tier(0.3)["name"] == "mild"


def test_get_text_overlap():
    text = get_text(55, 95)
    assert "sodium-potassium pump" in text and "rising phase" in text
    assert "Today we're covering" not in text


def test_prompt_contains_everything():
    prompt = build_prompt({"start": 90, "end": 120, "score": 0.1})
    assert "very low" in prompt and "rising phase" in prompt and "Write 3 sentences" in prompt
