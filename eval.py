from openai import OpenAI
import collections
from tqdm import tqdm
import json
from functools import partial
import logging
from concurrent.futures import ThreadPoolExecutor
import instructions_registry


class LLM:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
                base_url = config["LLM_API"],
                api_key=config["LLM_KEY"]
            ) 
        
    def get_response(self, or_data, sys_pmt):
        usr_prompt = or_data["prompt"]
        response = self.client.chat.completions.create(
        model=self.config["LLM_MODEL"],
        messages=[{"role": "system", "content": sys_pmt},
                {"role": "user", "content": usr_prompt}],
        **self.config["other parameters of llm"]
            )
        
        res = response.choices[0].message.content
        logger.info(f"system prompt:\n{sys_pmt}\n\nuser prompt:\n{usr_prompt}\n\nresponse:\n{res}")
        logger.info(f"-"*100)
        return res
    

class ModelEval:
    def __init__(self, llm:LLM, eval_criteria:str):
        self.model = llm
        self.eval_criteria = eval_criteria
    
    def load_dataset(self):
        data = []
        with open('./IFeval.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    
    
    def test_instruction_following_strict(self, inp, response):
        instruction_list = inp["instruction_id_list"]
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            instruction.build_description(**inp["kwargs"][index])
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=inp["prompt"])

            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        return {
            'instruction_id_list': inp["instruction_id_list"],
            'prompt': inp["prompt"],
            'response': response,
            'follow_all_instructions': all(is_following_list),
            'follow_instruction_list': is_following_list,
        }


    def test_instruction_following_loose(self, inp, response):
        r = response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        all_responses = [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]
        instruction_list = inp["instruction_id_list"]
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            instruction.build_description(**inp["kwargs"][index])
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=inp["prompt"])

            is_following = False
            for r in all_responses:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break

            is_following_list.append(is_following)

        return {
            "instruction_id_list": inp["instruction_id_list"],
            "prompt": inp["prompt"],
            "response": response,
            "follow_all_instructions": all(is_following_list),
            "follow_instruction_list": is_following_list,
        }
    
    def print_report(self, outputs):
        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        tier0_total = collections.defaultdict(int)
        tier0_correct = collections.defaultdict(int)

        tier1_total = collections.defaultdict(int)
        tier1_correct = collections.defaultdict(int)

        for example in outputs:
            follow_instruction_list = example["follow_instruction_list"]
            instruction_id_list = example["instruction_id_list"]

            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

            for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
                instruction_id = instruction_id.split(":")[0]
                tier0_total[instruction_id] += 1
                if followed_or_not:
                    tier0_correct[instruction_id] += 1

            for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
                tier1_total[instruction_id] += 1
                if followed_or_not:
                    tier1_correct[instruction_id] += 1

        print(f"prompt-level: {prompt_correct / prompt_total}")
        print(f"instruction-level: {instruction_correct / instruction_total}")
        print()
        for instruction_id in sorted(tier0_total.keys()):
            accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
            print(f"{instruction_id} {accuracy}")
        print()
        for instruction_id in sorted(tier1_total.keys()):
            accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
            print(f"{instruction_id} {accuracy}")

    def model_eval(self, system_prompt, max_workers=5):
        func = self.test_instruction_following_strict if self.eval_criteria == 'strict' else self.test_instruction_following_loose
        outputs = []
        or_data = self.load_dataset()
        partial_question_answer = partial(self.model.get_response, sys_pmt=system_prompt)
        len_data_set = len(or_data)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, res in enumerate(tqdm(executor.map(partial_question_answer, or_data), 
                                           total=len_data_set, 
                                           ncols=100,
                                           desc="Processing",
                                           unit='items',
                                           leave=True)):
                outputs.append(func(or_data[idx], res))
        # follow_all_instructions = [o["follow_all_instructions"] for o in outputs]
        # acc = sum(follow_all_instructions) / len_data_set
        # print(f"Acc:{acc}")
        self.print_report(outputs)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='IFEval')

    parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
    parser.add_argument('--eval_target', type=str, default='strict', choices=['strict', 'loose'], help='eval results strict or loose')
    

    args = parser.parse_args()

    log_path = f"./log.log"

    logger = logging.getLogger('IFEval')
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

    llm = LLM(config)
    llm_eval = ModelEval(llm, args.eval_target)
    llm_eval.model_eval(config["system prompt"], max_workers=args.num_workers)
    # print("The acc of "+ config["LLM_MODEL"]+ f" in MBPP dataset is {acc*100:.2f}%")
