"""
Checks the LLM's answers against your rules. Needs Ollama running.
Run with:  python test_llm.py
Each dip is sent several times because the model's answers vary.
"""
import re

from controller import ask_llm, build_prompt, get_tier, get_text, fake_session
from eeg_kernel import process_session

RUNS = 3
EXPECTED_SENTENCES = {"very low": 3, "low": 2, "mild": 1}


def count_sentences(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def made_up_numbers(answer, source):
    """Numbers in the answer that never appear in the transcript = likely hallucinated."""
    return set(re.findall(r"\d+", answer)) - set(re.findall(r"\d+", source))


def main():
    _, dips = process_session(*fake_session())
    passed = total = 0
    for dip in dips:
        tier = get_tier(dip["score"])["name"]
        source = get_text(max(0, dip["start"] - 20), dip["end"])
        print(f"\n===== {dip['start']}-{dip['end']}s  tier: {tier} =====")
        for i in range(RUNS):
            answer = ask_llm(build_prompt(dip))
            checks = {
                "starts right": answer.startswith("You may have missed"),
                "sentence count": count_sentences(answer) == EXPECTED_SENTENCES[tier],
                "no made-up numbers": not made_up_numbers(answer, source),
            }
            ok = all(checks.values())
            passed += ok
            total += 1
            print(f"\nRun {i + 1}: {'PASS' if ok else 'FAIL'}  {checks}")
            print(answer)
    print(f"\n{passed}/{total} answers passed every rule")


if __name__ == "__main__":
    main()
