

指令遵循能力数据集，IFEval数据集通过设计严格（Strict） 和 宽松（Loose）两种评估指标，更精准地衡量模型是否遵循给定指令。

原始数据集地址：https://huggingface.co/datasets/google/IFEval

github地址：https://github.com/google-research/google-research/tree/master/instruction_following_eval

数据集更详细的描述：https://blog.csdn.net/shizheng_Li/article/details/144516942

## 数据集key解释

- `key`：提示的唯一 ID。
- `prompt`：描述模型应该执行的任务。
- `instruction_id_list`：可验证指令数组。
- `kwargs`：用于指定中每个可验证指令的参数数组`instruction_id_list`。

## 数据集示例

```json
{
  "key": 1000, 
  
 "prompt": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.", 
  
  "instruction_id_list": ["punctuation:no_comma", "detectable_format:number_highlighted_sections", "length_constraints:number_words"], 
  
  "kwargs": [{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}]}
```

