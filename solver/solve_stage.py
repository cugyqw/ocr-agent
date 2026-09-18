"""阶段二：数学解题。

职责：拿 OCR 识别出的题目文本，输出解题过程。

模型是 Qwen2.5-Math-1.5B-Instruct —— 纯文本模型，
比 GLM-OCR 的 mrope 简单，标准的 generate 即可。
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# 系统提示：引导模型给出逐步推理。
# 注意 Qwen2.5-Math 官方建议用 "Please reason step by step,
# and put your final answer within \boxed{}."
SYSTEM_PROMPT = """你是一个数学解题助手。请根据用户提供的题目，给出清晰、严谨的解题过程。

要求：
1. 先理解题意，必要时复述题目
2. 逐步推理，每一步说明依据
3. 最终答案用 \\boxed{} 包裹
4. 使用中文作答，公式用 LaTeX

如果题目内容看起来不完整或识别有误，请先指出，再基于可读内容作答。"""


class SolveStage:
    """解题阶段：题目文本 → 解答。"""

    def __init__(
        self,
        model_path: str = "/home/yqw/桌面/ocr_agent/Qwen2.5-Math-1.5B-Instruct",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        """加载并常驻解题模型。"""
        logger.info("加载解题模型: %s", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        logger.info("解题阶段就绪")

    def run(self, problem: str, history: list[dict] | None = None) -> str:
        """解一道题。

        Args:
            problem: 题目文本（来自 OCR）
            history: 之前的多轮对话（可选），用于追问

        Returns:
            解题过程与答案
        """
        if self.model is None:
            raise RuntimeError("请先调用 load()")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": problem})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 显式构造生成参数，避免把 None 传给 generate 导致报错
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
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
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 只取新生成的部分
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        """释放显存。"""
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
