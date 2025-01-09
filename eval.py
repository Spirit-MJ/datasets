from openai import OpenAI
import numpy as np
import random
from tqdm import tqdm
import json
from functools import partial
import logging
from concurrent.futures import ThreadPoolExecutor

print("load config from ./config.json")
with open('./config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)


class LLM:
    def __init__(self, config, n):
        self.config = config
        self.n = n
        self.client = OpenAI(
                base_url = config["LLM_API"],
                api_key=config["LLM_KEY"]
            ) 
        
    def get_response(self, usr_prompt, sys_pmt):
        result = []
        response = self.client.chat.completions.create(
        model=self.config["LLM_MODEL"],
        messages=[{"role": "system", "content": sys_pmt},
                {"role": "user", "content": usr_prompt}],
        n = self.n,
        **self.config["other parameters of llm"]
            )
        
        for i in range(self.n):
            res = response.choices[i].message.content
            try:
                temp_res = res.split('```python\n')[1].split('```')[0].split('# Test')[0]
            except:
                temp_res = res
            logger.info(f"system prompt:\n{sys_pmt}\n\nuser prompt:\n{usr_prompt}\n\nresponse:\n{temp_res}")
            logger.info(f"-"*100)
            result.append(temp_res)
        return result
    

class ModelEval:
    def __init__(self, llm:LLM, data_name:str, few_shot_id:list, n:int, k:int):
        self.model = llm
        self.data_set_name=data_name
        self.few_shot_id = few_shot_id
        self.n = n
        self.k = k
    
    def load_dataset(self):
        user_prompt_ls, test_list, few_shot = [], [], []
        if self.data_set_name == "full":
            with open('./MBPP/full/test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    json_objects = json.loads(line)
                    user_prompt_ls.append(json_objects["text"]+"Your code should pass these tests:\n\n"+str(json_objects["test_list"]))
                    test_list.append(json_objects["test_list"])
            if isinstance(self.few_shot_id, list):
                with open('./MBPP/full/prompt.jsonl', 'r', encoding='utf-8') as file:
                    for idx, line in enumerate(file):
                        if (idx+1) in self.few_shot_id:
                            json_objects = json.loads(line)
                            few_shot.append((json_objects["text"], json_objects["code"], json_objects["test_list"]))
        elif self.data_set_name == "sanitized":
            with open('./MBPP/sanitized/test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    json_objects = json.loads(line)
                    user_prompt_ls.append(json_objects["prompt"]+"Your code should pass these tests:\n\n"+str(json_objects["test_list"]))
                    test_list.append(json_objects["test_list"])
            if isinstance(self.few_shot_id, list):
                with open('./MBPP/sanitized/prompt.jsonl', 'r', encoding='utf-8') as file:
                    for idx, line in enumerate(file):
                        if (idx+1) in self.few_shot_id:
                            json_objects = json.loads(line)
                            few_shot.append((json_objects["prompt"], json_objects["code"], json_objects["test_list"]))
        else:
            raise ValueError("data_set must be 'full' or 'sanitized'") 
        return user_prompt_ls[:2], test_list[:2], few_shot
    
    def get_system_prompt(self, few_shot_template:list):
        if few_shot_template:
            system_prompt = """You are an expert Python programmer, and here is your task:\n""" + \
            "".join([f"""{info[0]}Your code should pass these tests:\n\n{info[2]}\n[BEGIN]\n{info[1]}\n[DONE]\n""" for info in few_shot_template])
        else:
            system_prompt = "You are an expert Python programmer."
        return system_prompt
    
    def pass_at_k(self, response:list, test_case:list):

        """ 
        :param n: total number of samples 
        :param c: number of correct samples 
        :param k: k in pass@k
        """ 
        index = random.sample(range(self.n), self.k)
        c = 0
        for idx in index:
            try:
                exec(response[idx])
                tag = True
                for case in test_case:
                    try:
                        exec(case)
                    except:
                        tag = False
                        break
                if tag:
                    c += 1
            except:
                continue
        if self.n- c < self.k: 
            return 1.0 
        return 1.0- np.prod(1.0- self.k / np.arange(self.n- c + 1, self.n + 1))
    
    def model_eval(self, max_workers=5):
        acc = 0
        llm_input_ls, test_ls, few_shot = self.load_dataset()
        system_prompt = self.get_system_prompt(few_shot)
        partial_question_answer = partial(self.model.get_response, sys_pmt=system_prompt)
        len_data_set = len(llm_input_ls)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, res in enumerate(tqdm(executor.map(partial_question_answer, llm_input_ls), 
                                           total=len_data_set, 
                                           ncols=100,
                                           desc="Processing",
                                           unit='items',
                                           leave=True)):
                acc += self.pass_at_k(res, test_ls[idx])
        return acc / len_data_set


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='MBPP')

    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--data_set', type=str, default='full', help='full or sanitized')
    parser.add_argument('--n', type=int, default=5, help='total number of samples in pass@k')
    parser.add_argument('--k', type=int, default=2, help='k in pass@k')
    parser.add_argument('--few_shot_id', nargs='+', type=int, default=None, help='A list of integer IDs for few-shot learning.')

    args = parser.parse_args()

    log_path = f"./log.log"

    logger = logging.getLogger('MBPP')
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(log_path, encoding='utf-8') 
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)  

    logger.addHandler(fh)

    llm = LLM(config, args.n)
    llm_eval = ModelEval(llm, args.data_set, args.few_shot_id, args.n, args.k)
    acc = llm_eval.model_eval(max_workers=args.num_workers)
    print("The acc of "+ config["LLM_MODEL"]+ f" in MBPP dataset is {acc*100:.2f}%")
