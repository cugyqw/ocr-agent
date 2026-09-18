# 拍照解题系统（OCR + 数学推理）

基于 **GLM-OCR** 与 **Qwen2.5-Math** 的本地拍照解题流水线，并附带一个自研的轻量推理引擎 `miniinfer`。

```
图片 → [GLM-OCR 识图] → 文本(含 LaTeX) → [Qwen2.5-Math 解题] → 解答
```

全程本地推理，无需联网调用 API。

## 目录结构

```
.
├── solve_math.py        # 命令行入口
├── solver/              # 两阶段流水线
│   ├── pipeline.py      #   主控：串联 OCR 与解题
│   ├── ocr_stage.py     #   阶段一：GLM-OCR 识图
│   └── solve_stage.py   #   阶段二：Qwen2.5-Math 解题
├── miniinfer/           # 自研轻量推理引擎
│   ├── engine.py        #   引擎主循环（模型常驻 / 混合 prefill-decode）
│   ├── scheduler.py     #   连续批处理调度器
│   ├── kv_cache.py      #   KV Cache 合并拆分 + 前缀缓存
│   ├── sequence.py      #   请求状态机
│   ├── sampler.py       #   批量采样（greedy / top-k / top-p）
│   └── config.py        #   配置项
├── bench_miinfer.py     # 引擎性能基准
└── run.py               # 最简参考脚本
```

## 环境

- Python 3.12
- PyTorch 2.14 (CUDA 13.0)
- transformers 5.16
- 显卡：RTX 5060 Ti 8GB（实测两模型同时常驻峰值约 6~7GB）

依赖安装（清华源）：

```bash
pip install torch transformers accelerate pillow torchvision modelscope \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 注意：若 `triton` 报 `Python.h: No such file or directory`，需安装
> `sudo apt-get install -y python3.12-dev`

## 下载模型

模型权重未纳入版本管理，需自行下载（走 ModelScope，国内速度较好）：

```bash
# GLM-OCR（约 2.5GB）
modelscope download --model ZhipuAI/GLM-OCR --local_dir ./GLM-OCR

# Qwen2.5-Math-1.5B-Instruct（约 2.9GB）
modelscope download --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --local_dir ./Qwen2.5-Math-1.5B-Instruct
```

可用 `bash download_models.sh` 一键完成。

## 使用

```bash
# 单张图
python solve_math.py /path/to/problem.png

# 多张图（OCR 阶段自动批处理）
python solve_math.py q1.png q2.png q3.png

# 附加解题要求
python solve_math.py problem.png -i "只给出最终答案"

# 跳过公式识别（更快，但可能漏公式）
python solve_math.py problem.png --no-formula
```

输出包含三部分：OCR 文字、OCR 公式、解答过程。

## miniinfer 引擎

在 `transformers` 之上实现了 vLLM/SGLang 的核心机制，无额外依赖：

| 机制 | 对应实现 |
|------|----------|
| 连续批处理 | `scheduler.py` + `engine.py` 的 step 循环 |
| KV Cache 复用 | `kv_cache.py` 的 merge/split |
| 前缀缓存 | `kv_cache.py` 的 `PrefixCache` |
| 模型常驻 | `engine.py` 的 `load_model()` |
| CUDA 层 | 直接复用 PyTorch SDPA，不手写 kernel |

基准测试（4 条并发请求）：

```bash
python bench_miinfer.py
```

实测吞吐由 122.8 tok/s 提升至 202.3 tok/s（约 **1.65 倍**）。

### GLM-OCR 的关键约束

实测踩坑得出，已在代码注释中标注：

1. **decode 必须显式传 `position_ids`**
   GLM-OCR 使用 3D 位置编码（mrope，`mrope_section=[16,24,24]`），
   decode 阶段不传会触发张量维度不匹配错误。

2. **batch 推理必须传 `mm_token_type_ids`**
   该字段标记每个 token 属于文本(0)还是图片(1)。
   单条推理时可选，但 batch 场景下 mrope 依赖它区分模态，
   缺失会导致输出乱码。

3. **不同长度的序列无法合并成同一 batch**
   实测即使显式传入 `position_ids` 与 `mm_token_type_ids`，
   带 padding 的 batch 仍会算出错误结果。
   因此引擎采用**等长分桶**策略：仅合并 `total_len` 相同的序列。
   这与 vLLM 用 PagedAttention 任意拼装序列不同——后者需要模型侧配合。

## 已知局限

- **解题能力受模型规模限制**：1.5B 模型在计算题（方程、积分、面积）
  上表现良好，但在几何证明等需要严格多步推理的题目上容易出错，
  甚至出现循环论证。
- **OCR 双 prompt 各有噪声**：`Formula Recognition:` 会误处理非公式
  文本（如字序错乱），`Text Recognition:` 可能漏掉公式块。
- **选择题存在"强行选一个"倾向**：当计算结果与所有选项都不符时，
  模型可能不报告异常而直接选最接近的选项。

## License

代码部分仅供学习交流。模型版权归各自作者所有：
- [GLM-OCR](https://www.modelscope.cn/models/ZhipuAI/GLM-OCR) — MIT
- [Qwen2.5-Math](https://www.modelscope.cn/models/Qwen/Qwen2.5-Math-1.5B-Instruct) — Apache 2.0
