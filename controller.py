def get_tier(score):
    ''' 
    Takes score and evaluates tier for focus level. 
    Returns dictionary containing tier level and instruction for LLM.
    '''
    if score < 0.2:
        # returns a name and instruction for LLM --> dictionary
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

# example lecture transcript. times are in seconds.
# later this will come from real speech-to-text.
TRANSCRIPT = [
    {"start": 0,  "end": 30,  "text": "Today we're covering how neurons communicate. Resting membrane potential is about -70 millivolts."},
    {"start": 30, "end": 60,  "text": "When the neuron passes threshold, around -55 millivolts, voltage-gated sodium channels open."},
    {"start": 60, "end": 90,  "text": "Sodium rushes in and the membrane shoots up toward +40. That's the rising phase of the action potential."},
    {"start": 90, "end": 120, "text": "Then potassium channels open, potassium leaves, and the cell repolarizes."},
]

def get_text(start, end):
    '''
    Grabs text between two times. Returns list with transcript.
    '''
    pieces = []
    for segment in TRANSCRIPT:
        if segment["end"] > start and segment ["start"] < end:
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