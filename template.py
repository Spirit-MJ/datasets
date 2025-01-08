def get_system_prompt(few_shot_template:list):
    if few_shot_template:
        system_prompt = """You are an expert Python programmer, and here is your task:""" + \
        "".join([f"""{info[0]}Your code should pass these tests:\n\n{info[2]}\n[BEGIN]\n{info[1]}\n[DONE]""" for info in few_shot_template])
    else:
       system_prompt = "You are an expert Python programmer."
    return system_prompt