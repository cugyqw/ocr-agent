# 拍照解题系统（OCR + 多科目推理）

基于 **GLM-OCR** 与 **Qwen2.5** 的本地拍照解题流水线，面向**小学六年级以内**，
并附带一个自研的轻量推理引擎 `miniinfer`。

```
图片 → [GLM-OCR 识图] → 文本(含 LaTeX)
         ↓
    科目识别（数学/语文/英语）
         ↓
    数学 → Qwen2.5-Math-1.5B
    其他 → Qwen2.5-1.5B-Instruct
         ↓
       解答
```

全程本地推理，无需联网调用 API。

## 目录结构

```
.
├── solve_math.py        # 命令行入口
├── solver/              # 流水线
│   ├── pipeline.py      #   主控：串联 OCR → 科目识别 → 解题
│   ├── ocr_stage.py     #   阶段一：GLM-OCR 识图
│   ├── subject.py       #   科目识别（规则判断，零成本）
│   └── solve_stage.py   #   阶段二：多模型路由解题
├── miniinfer/           # 自研轻量推理引擎
│   ├── engine.py        #   引擎主循环（模型常驻 / 混合 prefill-decode）
│   ├── scheduler.py     #   连续批处理调度器
│   ├── kv_cache.py      #   KV Cache 合并拆分 + 前缀缓存
│   ├── sequence.py      #   请求状态机
│   ├── sampler.py       #   批量采样（greedy / top-k / top-p）
│   └── config.py        #   配置项
├── pdf_convert.py       # PDF 转 Markdown / Word 入口
├── pdfmd/               # PDF 处理模块
│   ├── converter.py     #   主控：自动判断类型并选择处理路径
│   ├── detector.py      #   类型检测（电子版 vs 扫描版）
│   ├── md_builder.py    #   Markdown 构建
│   ├── docx_builder.py  #   Word 构建
│   ├── text_rules.py    #   标题/公式判断规则（两个 builder 共用）
│   └── image_utils.py   #   图片预处理（裁剪留白）
├── tests/               # 测试
│   └── test_prefix_cache.py  # 前缀缓存正确性与收益验证
├── docs/                # 测试报告
│   ├── baseline_report.md    # 基线实测报告
│   └── eval_report.md        # CMMLU 评测报告
├── bench_miinfer.py     # 引擎性能基准
├── bench_baseline.py    # 解题能力基线测试
├── bench_clean.py       # CMMLU 数据集评测
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

# PDF 转换功能额外需要
pip install pdfplumber python-docx \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

PDF 功能还需要系统级的 `pdftoppm`（poppler-utils 提供）：

```bash
sudo apt-get install -y poppler-utils
```

> 注意：若 `triton` 报 `Python.h: No such file or directory`，需安装
> `sudo apt-get install -y python3.12-dev`

## 下载模型

模型权重未纳入版本管理，需自行下载（走 ModelScope，国内速度较好）：

```bash
# GLM-OCR（约 2.5GB）
modelscope download --model ZhipuAI/GLM-OCR --local_dir ./GLM-OCR

# Qwen2.5-Math-1.5B-Instruct（约 2.9GB）— 数学题专用
modelscope download --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --local_dir ./Qwen2.5-Math-1.5B-Instruct

# Qwen2.5-1.5B-Instruct（约 2.9GB）— 语文/英语通用
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct \
    --local_dir ./Qwen2.5-1.5B-Instruct
```

可用 `bash download_models.sh` 一键完成。

## 使用

```bash
# 单张图（自动识别科目并路由到对应模型）
python solve_math.py /path/to/problem.png

# 多张图（OCR 阶段自动批处理）
python solve_math.py q1.png q2.png q3.png

# 强制指定科目（跳过自动识别）
python solve_math.py problem.png --subject math

# 附加解题要求
python solve_math.py problem.png -i "只给出最终答案"

# 跳过公式识别（更快，但可能漏公式）
python solve_math.py problem.png --no-formula
```

输出包含四部分：识别出的科目、OCR 文字、OCR 公式、解答过程。

### 科目路由

`subject` 参数可选 `math` / `chinese` / `english` / `general`。

自动识别基于规则（`solver/subject.py`），零成本、零延迟：

- **数学** → Qwen2.5-Math-1.5B（数学专项微调）
- **语文 / 英语 / 其他** → Qwen2.5-1.5B-Instruct（通用）

两个解题模型**按需加载**，切换时自动卸载另一个以节省显存。

### 基线测试

```bash
# 测试通用版在小学题目上的表现
python bench_baseline.py --model general --out baseline_general.json

# 只测某一类: poem/literature/chinese/english/math
python bench_baseline.py --only poem
```

实测结果见 [基线测试报告](docs/baseline_report.md)。

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

### 前缀缓存

同一 prompt 重复请求时，直接复用已算好的 KV Cache，跳过 prefill。

```bash
python tests/test_prefix_cache.py   # 正确性与收益验证
```

实测加速（prompt 越长、输出越短，收益越大）：

| prompt 长度 | 输出长度 | 命中后耗时 | 加速 |
|---|---|---|---|
| 254 token | 8 | 0.06s | 1.2x |
| 1096 token | 4 | 0.04s | 1.9x |
| **1262 token** | **1** | **0.010s** | **5.5x** |

这正是 RAG 场景的形态——知识库内容注入 prompt 使 prompt 很长，
而答案很短。此时缓存收益最大。

**限制**：只对纯文本序列生效。多模态序列的 KV 依赖图片特征，
不同图片的 prompt token 可能相同但视觉内容不同，复用会导致错误。

`engine.full_stats()` 可查看命中情况：

```python
{'prefix_cache_hits': 3, 'cache_entries': 3, 'cache_hit_rate': 0.75, ...}
```

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

## PDF 转 Markdown / Word

自动判断 PDF 类型并选择处理路径：

```
PDF
 ├── 电子版 → pdfplumber 直接提取（快、准、零成本）
 ├── 扫描版 → pdftoppm 转图 → GLM-OCR
 └── 混合版 → 逐页判断
```

### 用法

```bash
# 单个/多个 PDF
python pdf_convert.py 报告.pdf
python pdf_convert.py a.pdf b.pdf -o ./output

# 批量处理目录
python pdf_convert.py ./pdfs/ -o ./output

# 只检测类型，不转换
python pdf_convert.py 报告.pdf --detect

# 只输出 Markdown
python pdf_convert.py 报告.pdf --no-docx

# 限制页数（大文档先试几页）
python pdf_convert.py 扫描件.pdf --max-pages 5
```

### 实测性能

| 类型 | 页数 | 耗时 | 说明 |
|---|---|---|---|
| 电子版 | 2 | **0.08s** | 直接提取，无需 OCR |
| 扫描版 | 1 | 1.3s | 含渲染 + OCR |
| 扫描版 | 2 | 5.7s | 含渲染 + OCR |

### 两个关键实现细节

**1. 渲染 DPI 必须控制在 150**

GLM-OCR 的图片 token 数约等于像素数 / 1000，上限 4096：

```
200 DPI -> A4 = 1654x2339 = 3.87MP -> 4968 token  ❌ 超限，静默失败
150 DPI -> A4 = 1240x1754 = 2.17MP -> 2784 token  ✅ 可用
```

用 200 DPI 时引擎会拒绝该请求，且**原本是静默失败**（返回空字符串无报错），
现已改为抛出明确异常。

**2. 必须裁剪页面留白**

扫描件渲染后常有大片空白（实测文字只占画面 4%），
会让 OCR 注意力稀释、识别不全。裁剪后同一页从"只识别出 1 行"
变为"4 行全对"，且耗时从 6.5s 降到 4.6s。

### 公式识别结果需要过滤

对没有公式的页面硬跑 `Formula Recognition:`，模型有两种错误行为：

1. 输出几百行空的 `$$` 标记
2. 把普通文本强行包进 `$$` / `\mathrm{}`（"伪公式"）

`_is_meaningful_formula()` 通过检查数学特征（运算符、LaTeX 数学命令、
矩阵结构）来过滤，单元测试 9/9 通过。

## 已知局限

### 模型能力

- **知识精度不足**：1.5B 模型存在"知道但不精确"的问题。实测发现
  李白被同时称为"诗仙"和"诗圣"（后者是杜甫）、"美丽→宝贵"、
  "节约→消耗"等错误。详见 [基线测试报告](docs/baseline_report.md)。
- **几何证明能力弱**：在需要严格多步推理的证明题上容易出错，
  出现过循环论证（把待证结论当作已知条件）。
- **选择题存在"强行选一个"倾向**：当计算结果与所有选项都不符时，
  模型可能不报告异常而直接选最接近的选项。

### OCR

- **两个 prompt 各有噪声**：`Formula Recognition:` 会误处理非公式
  文本（如字序错乱），`Text Recognition:` 可能漏掉公式块。
- **复杂排版会丢信息**：整图识别时可能漏掉题目的初始设定
  （如"设 C=0, B=1"这类关键条件）。

### 实测数据

| 场景 | 表现 |
|---|---|
| 小学数学题 | 全对 |
| 小学古诗默写 | 全对 |
| 小学语文基础（近反义词） | 67% |
| 小学英语 | 全对 |

## License

代码部分仅供学习交流。模型版权归各自作者所有：
- [GLM-OCR](https://www.modelscope.cn/models/ZhipuAI/GLM-OCR) — MIT
- [Qwen2.5-Math](https://www.modelscope.cn/models/Qwen/Qwen2.5-Math-1.5B-Instruct) — Apache 2.0
- [Qwen2.5](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) — Apache 2.0
