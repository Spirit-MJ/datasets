# 1.数据集概述

- AIME 2024 是2024年美国数学邀请赛，是面向美国高中生的顶级数学竞赛

- 数据集原始地址如下：[https://huggingface.co/datasets/Maxwell-Jia/AIME_2024?row=24](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024?row=24)

- 总共30道题数学题

# 2.数据示例
```json
{

"ID": "2024-II-1",

"Problem": "Among the $900$ residents of Aimeville, there are $195$ who own a diamond ring, $367$ who own a set of golf clubs, and $562$ who own a garden spade. In addition, each of the $900$ residents owns a bag of candy hearts. There are $437$ residents who own exactly two of these things, and $234$ residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.",

"Solution": "Let $w, x, y, z$ denote the number of residents who own 1, 2, 3, and 4 of these items, respectively. We know $w+x+y+z=900$, since there are 900 residents in total. This simplifies to $w+z=229$, since we know $x=437$ and $y=234$. Now, we set an equation for the total number of items. We know there are 195 rings, 367 clubs, 562 spades, and 900 candy hearts. Adding these up, there are 2024 items in total. Thus, $w+2x+3y+4z=2024$ since we are not adding the number of items each group of people contributes, and this must be equal to the total number of items. Plugging in $x$ and $y$, we get $w+4z=448$. Solving $w+z=229$ and $w+4z=448$, we get $z=73$.",

"Answer": "73"

}
```

- ID字段表示题的唯一标识
- Problem字段包含一个问题
- Solution字段表示问题参考解题过程
- answer字段表示针对该问题的参考答案。

# 3.文件说明
- [eval.py](eval.py)：评估脚本
- [run.sh](run.sh)：运行shell脚本
- [AIME_2024.jsonl](AIME_2024.jsonl)：原始数据
- [config.json](config.json)：大模型相关配置以及system prompt
- [log.log](log.log)：运行后的日志

# 4.使用[run.sh](run.sh)脚本进行评估，示例：

**注意要有相应的python环境！**

```shell
bash run.sh 0.1 \ # temperature
			20 \  # max_tokens
			10 \  # time_out
			50 \  # num-workers
```

# 5.使用[eval.py](eval.py)脚本进行评估，示例:

**注意要有相应的python环境！**

```shell
python eval.py --temperature 0.1 --max_tokens 20 --time_out 10 --num_workers 50
```

- 参数说明

```shell
--temperature temperature of LLM
--max_tokens  max_tokens of LLM
--time_out max response time of LLM
--num_workers number of workers
```

