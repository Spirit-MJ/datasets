# 1.数据集概述

- Arena-Hard-Auto 是一个自动化的指令调优大型语言模型（LLM）评估工具。它包含了**500**个具有挑战性的用户查询，并使用 GPT-4-Turbo 作为评判模型，将其他模型的响应与基准模型（默认是 GPT-4-0314）进行比较。
- 原始数据集地址：https://huggingface.co/datasets/lmarena-ai/arena-hard-auto-v0.1
- github地址：https://github.com/lmarena/arena-hard-auto

# 2.数据示例

```json
{
  "question_id": "328c149ed45a41c0b9d6f14659e63599", 
 
	"category": "arena-hard-v0.1", 
  
  "cluster": "ABC Sequence Puzzles & Groups", 
  
  "turns": [{"content": "Use ABC notation to write a melody in the style of a folk tune."}]

}
```

# 3.文件说明
- [eval.py](eval.py)：评估脚本
- [run.sh](run.sh)：运行shell脚本
- [Arena-Hard.jsonl](Arena-Hard.jsonl)：原始数据集
- [config.json](config.json)：大模型相关参数配置
- [requirements.txt](requirements.txt)：相应的python环境
- [log.log](log.log)：运行后产生的日志

# 4.使用[run.sh](run.sh)脚本进行评估，示例：

**注意要有相应的python环境！**

```shell
bash run.sh 16 # 并发数，默认为 16
```

# 5.使用[eval.py](eval.py)脚本进行评估，示例:

**注意要有相应的python环境！**

```shell
python eval.py --num_workers 16
```

- 参数说明

```shell
--num_workers  # number of workers
```
