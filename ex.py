import math

def calculate_elo(ra, rb, sa, sb, k=32):
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 / (1 + 10 ** ((ra - rb) / 400))
    ra_new = ra + k * (sa - ea)
    rb_new = rb + k * (sb - eb)
    return ra_new, rb_new

# 示例数据
models = ['ModelA', 'ModelB', 'ModelC']
elo_scores = {model: 1500 for model in models}
judgments = [
    ('ModelA', 'ModelB', 1),  # ModelA胜
    ('ModelB', 'ModelC', 1),  # ModelB胜
    ('ModelC', 'ModelA', 0),  # ModelA胜
]

# 更新Elo评分
for model1, model2, result in judgments:
    ra = elo_scores[model1]
    rb = elo_scores[model2]
    sa = result
    ra_new, rb_new = calculate_elo(ra, rb, sa)
    elo_scores[model1] = ra_new
    elo_scores[model2] = rb_new

# 输出最终Elo评分
for model, score in elo_scores.items():
    print(f"{model}: {score:.2f}")