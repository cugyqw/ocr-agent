"""解题阶段：根据科目选择模型并作答。

支持多个解题模型，按题型路由：
    - 数学题 → Qwen2.5-Math（数学专项，计算能力强）
    - 语文/英语/其他 → Qwen2.5 通用版

两个模型按需加载、用完可释放，避免显存超限。
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 按科目区分的 system prompt
# ---------------------------------------------------------------------------

SYSTEM_MATH = """你是一位小学数学老师，负责解答小学六年级以内的数学题。

要求：
1. 先读懂题目，明确已知条件和问题
2. 用小学生能理解的方式逐步讲解，步骤清晰
3. 计算要仔细，必要时验算
4. 最终答案要明确标出
5. 使用中文作答

如果题目信息不完整或识别有误，请先指出，再基于可读内容作答。"""

SYSTEM_CHINESE = """你是一位小学语文老师，负责解答小学六年级以内的语文题。

要求：
1. 根据题目要求作答，答案准确
2. 涉及古诗文时，要给出准确的原文、作者和朝代
3. 阅读理解题要结合原文分析，不要凭空发挥
4. 解答要符合小学生的理解水平，语言通俗
5. 使用中文作答

如果题目信息不完整或识别有误，请先指出，再基于可读内容作答。"""

SYSTEM_ENGLISH = """你是一位小学英语老师，负责解答小学六年级以内的英语题。

要求：
1. 给出准确的答案
2. 涉及语法时，简要说明规则（用中文解释，便于小学生理解）
3. 翻译要自然、通顺
4. 使用中文作答，英文部分保留英文

如果题目信息不完整或识别有误，请先指出，再基于可读内容作答。"""

SYSTEM_GENERAL = """你是一位小学老师，负责解答小学阶段的题目。

要求：
1. 准确理解题意，给出正确答案
2. 解答步骤清晰，符合小学生的理解水平
3. 使用中文作答

如果题目信息不完整或识别有误，请先指出，再基于可读内容作答。"""

# 科目 → system prompt
SUBJECT_PROMPTS = {
    "math": SYSTEM_MATH,
    "chinese": SYSTEM_CHINESE,
    "english": SYSTEM_ENGLISH,
    "general": SYSTEM_GENERAL,
}


class SolveStage:
    """解题阶段，支持多模型路由。"""

    def __init__(
        self,
        math_model_path: str = "/home/yqw/桌面/ocr_agent/Qwen2.5-Math-1.5B-Instruct",
        general_model_path: str = "/home/yqw/桌面/ocr_agent/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        auto_unload: bool = True,
    ):
        """
        Args:
            auto_unload: 切换模型时自动卸载另一个，节省显存。
                         显存充足可设为 False（切换更快）。
        """
        self.math_model_path = math_model_path
        self.general_model_path = general_model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.auto_unload = auto_unload

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 当前加载的模型：{"math" | "general": (model, tokenizer)}
        self._loaded: dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def _load(self, kind: str):
        """按需加载模型，返回 (model, tokenizer)。"""
        if kind in self._loaded:
            return self._loaded[kind]

        path = self.math_model_path if kind == "math" else self.general_model_path

        # 显存有限时先卸掉另一个
        if self.auto_unload and self._loaded:
            for other in list(self._loaded):
                logger.info("卸载模型 %s 腾出显存", other)
                self._unload_one(other)

        logger.info("加载解题模型[%s]: %s", kind, path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map=self.device
        )
        model.eval()
        self._loaded[kind] = (model, tokenizer)
        return model, tokenizer

    def _unload_one(self, kind: str) -> None:
        """卸载指定模型。"""
        if kind in self._loaded:
            del self._loaded[kind]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self) -> None:
        """就绪检查。不预热模型，等首次请求时按科目按需加载。

        之所以不预热：显存有限，而一次请求只会用到其中一个模型，
        预热反而会占着不需要的那份显存。
        """
        logger.info("解题阶段就绪（模型将按科目按需加载）")

    def unload(self) -> None:
        """释放全部模型。"""
        for kind in list(self._loaded):
            self._unload_one(kind)
        self._loaded.clear()

    # ------------------------------------------------------------------
    # 作答
    # ------------------------------------------------------------------
    def run(
        self,
        problem: str,
        subject: str = "general",
        history: list[dict] | None = None,
    ) -> str:
        """解一道题。

        Args:
            problem: 题目文本（来自 OCR）
            subject: 科目，决定用哪个模型和哪种提示词
                     "math" / "chinese" / "english" / "general"
            history: 多轮对话历史（可选）

        Returns:
            解答文本
        """
        # 模型选择：只有数学题用数学专项模型，其余用通用模型
        kind = "math" if subject == "math" else "general"
        model, tokenizer = self._load(kind)

        system_prompt = SUBJECT_PROMPTS.get(subject, SYSTEM_GENERAL)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": problem})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.temperature,
                top_p=0.8,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
