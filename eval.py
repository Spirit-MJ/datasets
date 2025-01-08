from openai import OpenAI
from tqdm import tqdm
import json
from functools import partial
from pydantic import BaseModel
import logging
from concurrent.futures import ThreadPoolExecutor

print("load config from ./config.json")
with open('./config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)


class ResponseType(BaseModel):
    response: int


class LLM:
    def __init__(self, json_schema, temperature, max_tokens, time_out):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.time_out = time_out
        self.json_schema = json_schema
        self.client = OpenAI(
                base_url = config["LLM_API"],
                api_key=config["LLM_KEY"]
            ) 
        
    def get_response(self, usr_prompt, sys_pmt):
        response = self.client.chat.completions.create(
        model=config["LLM_MODEL"],
        max_tokens=self.max_tokens,
        messages=[{"role": "system", "content": config["system_prompt"]},
                  {"role": "user", "content": usr_prompt}],
        temperature=self.temperature,
        timeout=self.time_out,
        response_format={
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": self.json_schema},
            } 
            )
        res = response.choices[0].message.content
        logger.info(f"system prompt:\n{sys_pmt}\n\nuser prompt:\n{usr_prompt}\n\nresponse:\n{str(res)}")
        logger.info(f"-"*100)
        return json.loads(res)["response"]
    

class ModelEval:
    def __init__(self, llm:LLM):
        self.model = llm
        self.data = self.load_dataset()
    
    def load_dataset(self):
        input_llm_data, ans = [], []
        with open('./AIME_2024.jsonl', 'r', encoding='utf-8') as file:
             for line in file:
                json_objects = json.loads(line)
                input_llm_data.append(json_objects["Problem"])
                ans.append(json_objects["Answer"])
        return input_llm_data, ans
    
    def model_eval(self, system_prompt:str, max_workers=5):
        correct_num = 0
        partial_question_answer = partial(self.model.get_response, sys_pmt=system_prompt)
        data_set, answer = self.data
        len_data_set = len(data_set)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, res in enumerate(tqdm(executor.map(partial_question_answer, data_set), 
                                           total=len_data_set, 
                                           ncols=100,
                                           desc="Processing",
                                           unit='items',
                                           leave=True)):
                if res == answer[idx]:
                    correct_num += 1
        return correct_num / len_data_set


if __name__ == "__main__":
    import argparse
 

    parser = argparse.ArgumentParser(description='AIME 2024')

    parser.add_argument('--temperature', type=float, default=0.5, help='temperature of LLM')
    parser.add_argument('--max_tokens', type=int, default=20, help='max_tokens of LLM')
    parser.add_argument('--time_out', type=float, default=10, help='max response time of LLM')
    parser.add_argument('--num_workers', type=int, default=20, help='number of workers')

    args = parser.parse_args()


    log_path = f"./log.log"

    logger = logging.getLogger('AIME_2024')
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(log_path, encoding='utf-8') 
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)  

    logger.addHandler(fh)

    json_schema = ResponseType.model_json_schema()
    llm = LLM(json_schema, args.temperature, args.max_tokens, args.time_out)
    llm_eval = ModelEval(llm)
    acc = llm_eval.model_eval(system_prompt=config["system_prompt"], max_workers=args.num_workers)
    print("The acc of "+ config["LLM_MODEL"]+ f" in AIME_2024 dataset is {acc*100:.2f}%")
