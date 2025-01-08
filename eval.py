from openai import OpenAI
from tqdm import tqdm
import json
from functools import partial
from pydantic import BaseModel
import logging
from concurrent.futures import ThreadPoolExecutor

print("load config from ./config.json")
with open('/Users/xiemingjiang/Pycharm程序/GitLab/dataset/llm-evaluation-datasets/config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)


class LLM:
    def __init__(self, temperature, max_tokens, time_out):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.time_out = time_out
        self.client = OpenAI(
                base_url = config["LLM_API"],
                api_key=config["LLM_KEY"]
            ) 
        
    def get_response(self, usr_prompt, sys_pmt):
        response = self.client.chat.completions.create(
        model=config["LLM_MODEL"],
        # max_tokens=self.max_tokens,
        messages=[{"role": "system", "content": sys_pmt},
                  {"role": "user", "content": usr_prompt}],
        temperature=self.temperature,
        timeout=self.time_out
            )
        res = response.choices[0].message.content
        logger.info(f"system prompt:\n{sys_pmt}\n\nuser prompt:\n{usr_prompt}\n\nresponse:\n{res}")
        logger.info(f"-"*100)
        return res
    

class ModelEval:
    def __init__(self, llm:LLM, data_name:str, few_shot_id:list):
        self.model = llm
        self.data_set_name=data_name
        self.few_shot_id = few_shot_id
    
    def load_dataset(self):
        user_prompt_ls, test_list, few_shot = [], [], []
        if self.data_set_name == "full":
            with open('/Users/xiemingjiang/Pycharm程序/GitLab/dataset/llm-evaluation-datasets/MBPP/full/test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    json_objects = json.loads(line)
                    user_prompt_ls.append(json_objects["text"]+"Your code should pass these tests:\n\n"+json_objects["test_list"])
                    test_list.append(json_objects["test_list"])
            if isinstance(self.few_shot_id, list):
                with open('/Users/xiemingjiang/Pycharm程序/GitLab/dataset/llm-evaluation-datasets/MBPP/full/prompt.jsonl', 'r', encoding='utf-8') as file:
                    for idx, line in enumerate(file):
                        if (idx+1) in self.few_shot_id:
                            json_objects = json.loads(line)
                            few_shot.append((json_objects["text"], json_objects["code"], json_objects["test_list"]))
        elif self.data_set_name == "sanitized":
            with open('/Users/xiemingjiang/Pycharm程序/GitLab/dataset/llm-evaluation-datasets/MBPP/sanitized/test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    json_objects = json.loads(line)
                    user_prompt_ls.append(json_objects["prompt"]+"Your code should pass these tests:\n\n"+json_objects["test_list"])
                    test_list.append(json_objects["test_list"])
            if isinstance(self.few_shot_id, list):
                with open('/Users/xiemingjiang/Pycharm程序/GitLab/dataset/llm-evaluation-datasets/MBPP/sanitized/prompt.jsonl', 'r', encoding='utf-8') as file:
                    for idx, line in enumerate(file):
                        if (idx+1) in self.few_shot_id:
                            json_objects = json.loads(line)
                            few_shot.append((json_objects["prompt"], json_objects["code"], json_objects["test_list"]))
        else:
            raise ValueError("data_set must be 'full' or 'sanitized'") 
        return user_prompt_ls, test_list, few_shot
    
    def get_system_prompt(self, few_shot_template:list):
        if few_shot_template:
            system_prompt = """You are an expert Python programmer, and here is your task:\n""" + \
            "".join([f"""{info[0]}Your code should pass these tests:\n\n{info[2]}\n[BEGIN]\n{info[1]}\n[DONE]\n""" for info in few_shot_template])
        else:
            system_prompt = "You are an expert Python programmer."
        return system_prompt
    
    def model_eval(self, max_workers=5):
        correct_num = 0
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
                pass
        return correct_num / len_data_set


if __name__ == "__main__":
    import argparse
 

    parser = argparse.ArgumentParser(description='MBPP')

    parser.add_argument('--temperature', type=float, default=0.5, help='temperature of LLM')
    parser.add_argument('--max_tokens', type=int, default=10000, help='max_tokens of LLM')
    parser.add_argument('--time_out', type=float, default=30, help='max response time of LLM')
    parser.add_argument('--num_workers', type=int, default=50, help='number of workers')
    parser.add_argument('--data_set', type=str, default='full', help='full or sanitized')
    parser.add_argument('--few_shot_id', type=list[int], default=[2, 3, 4], help='full or sanitized')

    args = parser.parse_args()


    log_path = f"/Users/xiemingjiang/Pycharm程序/GitLab/dataset/llm-evaluation-datasets/log.log"

    logger = logging.getLogger('MBPP')
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(log_path, encoding='utf-8') 
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)  

    logger.addHandler(fh)

    llm = LLM(args.temperature, args.max_tokens, args.time_out)
    llm_eval = ModelEval(llm, args.data_set, args.few_shot_id)
    acc = llm_eval.model_eval(max_workers=args.num_workers)
    print("The acc of "+ config["LLM_MODEL"]+ f" in MBPP dataset is {acc*100:.2f}%")
