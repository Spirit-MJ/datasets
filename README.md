# 1.数据集概述

- 指令遵循能力数据集，总共有**541**个测试样本，IFEval数据集通过设计严格（Strict） 和 宽松（Loose）两种评估指标，更精准地衡量模型是否遵循给定指令
- 原始数据集地址：https://huggingface.co/datasets/google/IFEval
- github地址：https://github.com/google-research/google-research/tree/master/instruction_following_eval
- 数据集更详细的描述：https://blog.csdn.net/shizheng_Li/article/details/144516942

# 2.数据示例

```json
{
  "key": 1000, 
  
 "prompt": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.", 
  
  "instruction_id_list": ["punctuation:no_comma", "detectable_format:number_highlighted_sections", "length_constraints:number_words"], 
  
  "kwargs": [{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}]
}
```

**数据集解释**

- `key`：提示的唯一 ID。
- `prompt`：描述模型应该执行的任务。
- `instruction_id_list`：可验证指令数组。
- `kwargs`：用于指定中每个可验证指令的参数数组`instruction_id_list`

# 3.文件说明
- [eval.py](eval.py)：评估脚本
- [run.sh](run.sh)：运行shell脚本
- [IFeval](IFeval)：原始数据集
- [config.json](config.json)：大模型相关参数配置
- [requirements.txt](requirements.txt)：相应的python环境
- [log.log](log.log)：运行后产生的日志

# 4.使用[run.sh](run.sh)脚本进行评估，示例：

**注意要有相应的python环境！**

```shell
bash run.sh strict \ # 使用严格的指标进行计算， 默认为 'strict'
        16 \  # 并发数，默认为 16
```

# 5.使用[eval.py](eval.py)脚本进行评估，示例:

**注意要有相应的python环境！**

```shell
python eval.py --eval_target strict --num_workers 16
```

- 参数说明

```shell
--eval_target  # eval results strict or loose
--num_workers  # number of workers
```
