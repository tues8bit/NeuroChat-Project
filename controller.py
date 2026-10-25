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


# test
tier = get_tier(0.15)
print(tier["name"])
print(tier["instruction"])