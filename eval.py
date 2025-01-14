from openai import OpenAI
from tqdm import tqdm
import json
from functools import partial
from pydantic import BaseModel
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor


class SelectStr(str, Enum):
    a = "[[A>>B]]"
    b = "[[A>B]]"
    c = "[[A=B]]"
    d = "[[B>A]]"
    e = "[[B>>A]]"


class ResponseType(BaseModel):
    response: SelectStr


class LLM:
    def __init__(self, config, json_schema=False):
        self.config = config
        self.json_schema = json_schema
        self.client = OpenAI(
                base_url = config["LLM_API"],
                api_key=config["LLM_KEY"]
            ) 
        
    def get_response(self, usr_prompt, sys_pmt):
        response = self.client.chat.completions.create(
                model=self.config["LLM_MODEL"],
                messages=[{"role": "system", "content": sys_pmt},
                        {"role": "user", "content": usr_prompt}],
                response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "foo", "schema": self.json_schema},
                                } if self.json_schema else None ,
                **self.config["other parameters of llm"] 
            )
        res = response.choices[0].message.content
        logger.info(f"system prompt:\n{sys_pmt}\n\nuser prompt:\n{usr_prompt}\n\nresponse:\n{str(res)}")
        logger.info(f"-"*150)
        if self.json_schema:
            res = json.loads(res)["response"]
        return res
    

class ModelEval:
    def __init__(self, llm:LLM, judge=False):
        self.model = llm
        self.judge = judge
    
    def load_dataset(self):
        or_data = []
        with open('./Arena-Hard.jsonl', 'r', encoding='utf-8') as file:
             for line in file:
                json_objects = json.loads(line)
                or_data.append(json_objects["turns"][0]["content"])
        return or_data[:2]
    
    def model_eval(self, system_prompt:str, max_workers=5):
        result = {}
        partial_question_answer = partial(self.model.get_response, sys_pmt=system_prompt)
        if self.judge:
            inputs = self.judge
        else:
            inputs = self.load_dataset()
        len_data_set = len(inputs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, res in enumerate(tqdm(executor.map(partial_question_answer, inputs), 
                                           total=len_data_set, 
                                           ncols=100,
                                           desc="Processing",
                                           unit='items',
                                           leave=True)):
                result[inputs[idx]] = res
        return result


def get_score(outputs):
    a_far_exceeds_b, a_exceeds_b = 0, 0
    a_equals_b = 0
    b_exceeds_a, b_far_exceeds_a = 0, 0
    for item in outputs:
        if item == "[[A>>B]]":
            a_far_exceeds_b += 1
        elif item == "[[A>B]]":
            a_exceeds_b += 1
        elif item == "[[A=B]]":
            a_equals_b += 1
        elif item == "[[B>A]]":
            b_exceeds_a += 1
        else:
            b_far_exceeds_a += 1
    return {"[[A>>B]]": a_far_exceeds_b,
            "[[A>B]]": a_exceeds_b, 
            "[[A=B]]": a_equals_b,
            "[[B>A]]": b_exceeds_a,
            "[[B>>A]]": b_far_exceeds_a}


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Arena')

    parser.add_argument('--num_workers', type=int, default=32, help='number of workers')

    args = parser.parse_args()


    log_path = f"./log.log"

    logger = logging.getLogger('Arena')
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(log_path, encoding='utf-8') 
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)  

    logger.addHandler(fh)

    print("load config from ./config.json")
    with open('./config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    get_ans_llm, baseline_llm = LLM(config["get_ans_model"]), LLM(config["baseline_model"])
    get_ans_llm_eval = ModelEval(get_ans_llm)
    baseline_llm_eval = ModelEval(baseline_llm)
    get_ans_result = get_ans_llm_eval.model_eval(system_prompt=config["get_ans_model"]["system_prompt"], max_workers=args.num_workers)
    baseline_result = baseline_llm_eval.model_eval(system_prompt=config["baseline_model"]["system_prompt"], max_workers=args.num_workers)
    judge_input = []
    for k, v in get_ans_result.items():
        judge_input.append("<|Question|>\n" + k + "\n\n<|The Start of Assistant A's Answer|>\n" + v + \
                           "\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n" + \
                            baseline_result[k] + "<|The End of Assistant B's Answer|>") 
    
    json_schema = ResponseType.model_json_schema()
    judge_llm = LLM(config["judge_model"], json_schema)
    judge_model = ModelEval(judge_llm, judge_input)
    judge_response = judge_model.model_eval(system_prompt=config["judge_model"]["system_prompt"], max_workers=args.num_workers)
    judge_result = []
    for v in judge_response.values():
        judge_result.append(v)
    results_dict = get_score(judge_result)
    print(results_dict)
