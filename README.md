# 1.数据集概述

- MBPP旨在评估和提升编程模型在基础Python编程任务上的表现。该数据集包含***\*974\****个编程任务，这些任务由入门级程序员设计，旨在通过自然语言描述来合成简短的Python程序。每个任务都包括一个具体问题的描述、一个解决该问题的Python函数，以及三个用于验证函数正确性的测试用例。这些测试用例以断言（assert）语句的形式编写，确保代码在执行时的正确性。
- MBPP数据集有两个版本：完整版、编辑版，后者通过手动检查和编辑，确保了问题的清晰度和测试用例的准确性。
- 数据集用法：[https://github.com/google-research/google-research/blob/master/mbpp/README.md](https://github.com/google-research/google-research/blob/master/mbpp/README.md)
- 数据集原始地址如下：[https://huggingface.co/datasets/google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp)

# 2.数据集划分

- 数据集数量：full:**974**   sanitized:**427**（full为完整版，santized为编辑版本，）

|           | **train**(训练) | **validation(微调时验证)** | **test(测试)** | **prompt(few shot)** |
| :-------: | :-------------: | :------------------------: | :------------: | :------------------: |
|   full    |       374       |             90             |      500       |          10          |
| sanitized |       120       |             43             |      257       |          7           |

# 3.数据示例

## 3.1 full数据集示例

```json
{
    'task_id': 1,
  
    'text': 'Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].',
  
    'code': 'R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]',
  
    'test_list': [
        'assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8',
        'assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12',
        'assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16'],
    'test_setup_code': '',
  
    'challenge_test_list': []
}
```

## 3.2 sanitized数据集示例

```json
{
    'source_file': 'Benchmark Questions Verification V2.ipynb',
  
    'task_id': 2,
  
    'prompt': 'Write a function to find the shared elements from the given two lists.',
  
    'code': 'def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ',
  
    'test_imports': [],
  
    'test_list': [
        'assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))',
        'assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))',
        'assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))'
        ]
}
```

# 4.文件说明
- [eval.py](eval.py)：评估脚本
- [run.sh](run.sh)：运行shell脚本
- [MPBB](MPBB)：数据集分为[完整版](full)以及[编辑版](sanitized)，每个数据集又分为[训练集](train.jsonl)、[验证集](validation.jsonl)、[测试集](test.jsonl)以及[few shot](prompt.jsonl)
- [config.json](config.json)：大模型相关参数配置
- [requirements.txt](requirements.txt)：需要的python环境
- [log.log](log.log)：运行后产生的日志

# 5.使用[run.sh](run.sh)脚本进行评估，示例：

**注意要有相应的python环境！**

```shell
bash run.sh full \ # full数据集or sanitized数据集， 默认为 'full'
        5 \  # total number of samples in pass@k， 默认为 5
        1 \  # k in pass@k， 默认为 1
        2 \  # 并发数，默认为 1
        2,3,4  # 选取id为哪些的数据作为few shot，例如 2,3,4， 可选参数 
```

# 6.使用[eval.py](eval.py)脚本进行评估，示例:

**注意要有相应的python环境！**

```shell
python eval.py --num_workers 2 --data_set full --n 5 --k 2 --few_shot_id 1 2 3
```

- 参数说明

```shell
--num_workers  # number of workers
--data_set  # full or sanitized
--n  # total number of samples in pass@k
--k  # k in pass@k
--few_shot_id  # A list of integer IDs for few-shot learning
```
