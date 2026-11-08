"""
Tests that don't need Ollama. Run with:  pytest -v
"""
import numpy as np
 
from eeg_kernel import (FS, make_fake_eeg, clean, engagement_index,
                        normalize, smooth, find_dips, process_session)
from controller import get_tier, get_text, build_prompt, fake_session
 
 
# ---------------- kernel: does the math behave like the brain should?
 
def test_focused_scores_higher_than_zoned_out():
    _, E_focus = engagement_index(clean(make_fake_eeg([1.0] * 30)))
    _, E_zoned = engagement_index(clean(make_fake_eeg([0.0] * 30)))
    assert E_focus.mean() > E_zoned.mean()
 
 
def test_filter_removes_60hz_hum():
    t = np.arange(10 * FS) / FS
    hum = np.sin(2 * np.pi * 60 * t)
    cleaned = clean(hum)[FS:-FS]                 # skip 1 s at each edge (filters wobble there)
    assert np.abs(cleaned).max() < 0.05          # basically gone
 
 
def test_epochs_every_quarter_second():
    times, E = engagement_index(np.random.randn(10 * FS))
    assert len(E) == 37                          # (10 s - 1 s window) / 0.25 s + 1
    assert np.allclose(np.diff(times), 0.25)
 
 
def test_normalize_stays_between_0_and_1():
    out = normalize(np.array([-5, 0.5, 99]), 0, 1)
    assert out.min() >= 0 and out.max() <= 1
 
 
def test_smooth_gives_one_score_per_second():
    times = np.arange(1, 61, 0.25)
    scores = smooth(times, np.ones_like(times))
    assert len(scores) == 60 and np.allclose(scores, 1)
 
 
# ---------------- dip detection edge cases
 
def test_dip_exactly_15_seconds_counts():
    assert len(find_dips([0.9] * 10 + [0.1] * 15 + [0.9] * 10)) == 1
 
 
def test_dip_14_seconds_does_not_count():
    assert find_dips([0.9] * 10 + [0.1] * 14 + [0.9] * 10) == []
 
 
def test_dip_at_very_end_is_caught():
    dips = find_dips([0.9] * 10 + [0.1] * 20)
    assert dips == [{"start": 10, "end": 30, "score": 0.1}]
 
 
def test_no_dips_when_always_focused():
    assert find_dips([0.9] * 100) == []
 
 
# ---------------- full pipeline on the fake lecture
 
def test_finds_all_three_fake_zone_outs():
    _, dips = process_session(*fake_session())
    assert len(dips) == 3
    # each detected dip should overlap the real zone-out (smoothing makes it a bit late)
    for dip, (real_start, real_end) in zip(dips, [(40, 80), (130, 165), (210, 245)]):
        assert dip["start"] < real_end and dip["end"] > real_start
 
 
def test_three_dips_land_in_three_different_tiers():
    _, dips = process_session(*fake_session())
    tiers = [get_tier(d["score"])["name"] for d in dips]
    assert tiers == ["very low", "low", "mild"]
 
 
# ---------------- controller
 
def test_tier_boundaries():
    assert get_tier(0.0)["name"] == "very low"
    assert get_tier(0.149)["name"] == "very low"
    assert get_tier(0.15)["name"] == "low"
    assert get_tier(0.25)["name"] == "mild"
 
 
def test_get_text_overlap():
    text = get_text(55, 95)
    assert "sodium-potassium pump" in text and "rising phase" in text
    assert "Today we're covering" not in text
 
 
def test_prompt_contains_everything():
    prompt = build_prompt({"start": 90, "end": 120, "score": 0.1})
    assert "very low" in prompt
    assert "rising phase" in prompt
    assert "Write 3 sentences" in prompt
 
