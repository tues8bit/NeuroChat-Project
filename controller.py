import requests

from eeg_kernel import load_muse_csv, process_session, save_scores


def get_tier(score):
    '''
    Takes score and evaluates tier for focus level.
    Returns dictionary containing tier level and instruction for LLM.
    '''
    if score < 0.2:
        return {
            "name": "very low",
            "instruction": "The listener likely missed this completely. "
                           "Write 3 sentences: re-explain the main idea simply, "
                           "then give the one key fact to remember."
        }
    elif score < 0.3:
        return {
            "name": "low",
            "instruction": "The listener caught part of this. "
                           "Write 2 sentences covering what they most likely missed."
        }
    else:
        return {
            "name": "mild",
            "instruction": "The listener was a little distracted. "
                           "Write 1 sentence as a quick reminder."
        }


# example lecture transcript, 5 minutes. times are in seconds.
# later this will come from real speech-to-text.
TRANSCRIPT = [
    {"start": 0,   "end": 30,  "text": "Today we're covering how neurons communicate. Resting membrane potential is about -70 millivolts."},
    {"start": 30,  "end": 60,  "text": "The sodium-potassium pump keeps it there by moving 3 sodium ions out and 2 potassium ions in."},
    {"start": 60,  "end": 90,  "text": "When the neuron passes threshold, around -55 millivolts, voltage-gated sodium channels open."},
    {"start": 90,  "end": 120, "text": "Sodium rushes in and the membrane shoots up toward +40. That's the rising phase of the action potential."},
    {"start": 120, "end": 150, "text": "Then potassium channels open, potassium leaves, and the cell repolarizes."},
    {"start": 150, "end": 180, "text": "It overshoots below rest, called hyperpolarization, which helps cause the refractory period."},
    {"start": 180, "end": 210, "text": "During the absolute refractory period, no new spike can fire no matter how strong the stimulus."},
    {"start": 210, "end": 240, "text": "Myelin insulates the axon so the signal jumps between nodes of Ranvier. This is saltatory conduction."},
    {"start": 240, "end": 270, "text": "At the axon terminal, calcium enters and triggers release of neurotransmitter into the synapse."},
    {"start": 270, "end": 300, "text": "The neurotransmitter binds receptors on the next neuron, and the process starts again."},
]


def get_text(start, end):
    '''
    Grabs text between two times. Returns one string of transcript.
    '''
    pieces = []
    for segment in TRANSCRIPT:
        if segment["end"] > start and segment["start"] < end:
            pieces.append(segment["text"])
    return " ".join(pieces)


def build_prompt(dip):
    '''
    Takes dip and combines tier instructions and transcript
    into one message for the LLM.
    '''
    tier = get_tier(dip["score"])
    context = get_text(max(0, dip["start"] - 20), dip["start"])
    missed = get_text(dip["start"], dip["end"])

    prompt = f"""Attention level: {tier["name"]} (score {dip["score"]} out of 1)

Context just before:
{context}

What was said while attention was low:
{missed}

Instructions: {tier["instruction"]}"""

    return prompt


SYSTEM_PROMPT = """You help a student catch up on parts of a lecture they missed.
An EEG headband measured their attention. You will get one moment where attention
was low, the transcript from that moment, and a little context from just before it.

Rules:
- Use ONLY information in the transcript. Never add facts that are not there.
- Write in second person, starting with "You may have missed".
- Follow the instruction's sentence count exactly.
- No preamble or headings. Just the summary."""


def ask_llm(user_prompt):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3.5:2b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3},
        },
    )
    return response.json()["message"]["content"]


def load_session(folder="data"):
    '''
    Reads the 3 recordings from CSV files:
    the lecture, plus the relax and task calibration recordings.
    To use real data, put real files with these names in the folder.
    '''
    lecture = load_muse_csv(f"{folder}/lecture.csv")
    relax = load_muse_csv(f"{folder}/relax.csv")
    task = load_muse_csv(f"{folder}/task.csv")
    return lecture, relax, task


if __name__ == "__main__":
    rows, scores, dips = process_session(*load_session())
    save_scores("data/scores.csv", rows)
    print("Saved theta/alpha/beta/engagement for every window to data/scores.csv")
    print(f"Found {len(dips)} dips\n")
    for dip in dips:
        print(f"--- {dip['start']}s to {dip['end']}s, score {dip['score']}, "
              f"tier: {get_tier(dip['score'])['name']} ---")
        print(ask_llm(build_prompt(dip)), "\n")
