- AIME 2024 是2024年美国数学邀请赛，是面向美国高中生的顶级数学竞赛

- `https://huggingface.co/datasets/Maxwell-Jia/AIME_2024?row=24`

- 总共30道题

- 数据格式如下：

```json
{

"ID": "2024-II-1",

"Problem": "Among the $900$ residents of Aimeville, there are $195$ who own a diamond ring, $367$ who own a set of golf clubs, and $562$ who own a garden spade. In addition, each of the $900$ residents owns a bag of candy hearts. There are $437$ residents who own exactly two of these things, and $234$ residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.",

"Solution": "Let $w, x, y, z$ denote the number of residents who own 1, 2, 3, and 4 of these items, respectively. We know $w+x+y+z=900$, since there are 900 residents in total. This simplifies to $w+z=229$, since we know $x=437$ and $y=234$. Now, we set an equation for the total number of items. We know there are 195 rings, 367 clubs, 562 spades, and 900 candy hearts. Adding these up, there are 2024 items in total. Thus, $w+2x+3y+4z=2024$ since we are not adding the number of items each group of people contributes, and this must be equal to the total number of items. Plugging in $x$ and $y$, we get $w+4z=448$. Solving $w+z=229$ and $w+4z=448$, we get $z=73$.",

"Answer": "73"

}
```