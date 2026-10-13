#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_run.py - 本地伪装模型运行器（支持 Ollama）
用于在 Mac 本地（无 MindSpore / NPU 环境）运行 LMCC T1 例题评测。

原理：
- 保持 submission.py 和 evaluate.py 完全官方原样
- 通过 monkey-patch 将 AutoModelForCausalLM / AutoTokenizer 替换为本地实现
- 默认通过 Ollama API 调用你本地部署的模型（http://localhost:11434）
- 也可回退到纯伪装模式（返回期望答案）用于调试流程

使用方法：
1. 确保 Ollama 正在运行，并且已 pull 对应模型（例如 qwen3:0.6b 或你实际使用的名字）
2. 目录结构：
   your_project/
   ├── evaluate.py          # 官方原样
   ├── submission.py        # 官方原样
   ├── local_run.py         # 本文件
   └── ../data/test_data_T1.jsonl

3. 运行：
   # 默认使用 Ollama（模型名可通过环境变量指定）
   OLLAMA_MODEL=qwen3:0.6b python local_run.py

   # 或纯伪装模式（不调用真实模型）
   USE_FAKE=1 python local_run.py

   python local_run.py --mode grading

注意：本文件仅用于本地调试，不提交到评测系统。
"""

import os
import sys
import types
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional

# ============================================================
# 0. Mock torch（如果环境没有安装）
# ============================================================
try:
    import torch
except ImportError:
    print("[local_run] 未检测到 torch，正在创建伪装 torch ...")
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.long = "long"
    torch.Tensor = type("Tensor", (), {})
    torch.device = lambda x: types.SimpleNamespace(type="cpu")
    torch.manual_seed = lambda x: None
    class NoGrad:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            return False
    torch.no_grad = lambda: NoGrad()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.npu = types.SimpleNamespace(is_available=lambda: False)
    
    class FakeTensor:
        def __init__(self, shape, dtype=None):
            self.shape = shape
            self.dtype = dtype
        def to(self, device):
            return self
        def __getitem__(self, idx):
            return self
        def __len__(self):
            return self.shape[-1] if self.shape else 1
    
    def zeros(shape, dtype=None):
        return FakeTensor(shape, dtype)
    torch.zeros = zeros
    torch.ones = zeros
    sys.modules["torch"] = torch
    print("[local_run] torch 伪装完成")

# ============================================================
# 1. Mock mindnlp + transformers（Mac 本地通常没有）
# ============================================================
if "mindnlp" not in sys.modules:
    mindnlp_mock = types.ModuleType("mindnlp")
    sys.modules["mindnlp"] = mindnlp_mock
    print("[local_run] 已伪装 mindnlp 模块")

try:
    import transformers
except ImportError:
    print("[local_run] 未检测到 transformers，正在创建伪装 ...")
    transformers = types.ModuleType("transformers")
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    sys.modules["transformers"] = transformers
    print("[local_run] transformers 伪装完成")


# ============================================================
# Ollama 配置
# ============================================================
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")  # 改成你实际的模型名
USE_FAKE = os.environ.get("USE_FAKE", "0") == "1"


def call_ollama(messages: List[Dict], generation_params: Dict) -> str:
    """调用 Ollama /api/chat，返回模型生成的文本"""
    url = f"{OLLAMA_HOST}/api/chat"
    
    # 映射生成参数
    options = {}
    if "max_new_tokens" in generation_params:
        options["num_predict"] = generation_params["max_new_tokens"]
    if generation_params.get("do_sample") is False:
        options["temperature"] = 0
    elif "temperature" in generation_params:
        options["temperature"] = generation_params["temperature"]
    if "top_p" in generation_params:
        options["top_p"] = generation_params["top_p"]
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    
    # enable_thinking 支持（部分模型）
    if generation_params.get("enable_thinking") is False:
        payload["think"] = False
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("message", {}).get("content", "")
            # 有些模型会把 thinking 放在单独字段
            thinking = result.get("message", {}).get("thinking", "")
            if thinking and generation_params.get("enable_thinking"):
                return f"<think>{thinking}</think>\n{content}"
            return content
    except Exception as e:
        print(f"[local_run] Ollama 调用失败: {e}")
        return '{"student_id": "error", "judgements": [false, false, false, false, false]}'


# ============================================================
# 2. 读取测试数据（仅纯伪装模式使用）
# ============================================================
def load_expected_answers() -> Dict[str, List[bool]]:
    candidates = [
        Path(__file__).parent.parent / "data" / "test_data_T1.jsonl",
        Path(__file__).parent / "data" / "test_data_T1.jsonl",
        Path.cwd() / "data" / "test_data_T1.jsonl",
        Path.cwd().parent / "data" / "test_data_T1.jsonl",
    ]
    for p in candidates:
        if p.exists():
            answers = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        answers[item["student_id"]] = item["expected_judgements"]
            print(f"[local_run] 已加载期望答案：{p}")
            return answers
    return {}


EXPECTED_ANSWERS = load_expected_answers() if USE_FAKE else {}


# ============================================================
# 3. 伪装 Tokenizer（缓存 messages 供 Ollama 使用）
# ============================================================
class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 151645
        self.pad_token_id = 151643
        self.chat_template = None
        self._last_rendered = ""
        self._last_messages: List[Dict] = []
        self._last_output = ""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kwargs):
        # 缓存原始 messages，供 Ollama /api/chat 使用
        self._last_messages = messages
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\nassistant:"
        self._last_rendered = text
        return text

    def encode(self, text, add_special_tokens=False, **kwargs):
        return list(range(min(len(text), 200)))

    def __call__(self, text, return_tensors="pt", padding=True, **kwargs):
        length = min(len(text), 128)
        input_ids = torch.zeros((1, length), dtype=torch.long)
        attention_mask = torch.zeros((1, length), dtype=torch.long)
        
        class Batch:
            def __init__(self):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
            def to(self, device):
                return self
            def __getitem__(self, key):
                return getattr(self, key)
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        return Batch()

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return self._last_output


# ============================================================
# 4. 伪装 Model（调用 Ollama 或纯伪装）
# ============================================================
class FakeModel:
    def __init__(self, tokenizer: FakeTokenizer):
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self._generation_params: Dict = {}

    def eval(self):
        return self

    def parameters(self):
        class FakeParam:
            device = torch.device("cpu")
            def numel(self):
                return 600_000_000
            requires_grad = False
        yield FakeParam()

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        # 从 kwargs 中提取生成参数（evaluate.py 会把它们传进来）
        gen_params = {k: v for k, v in kwargs.items() if k not in ("input_ids", "attention_mask", "eos_token_id", "pad_token_id")}
        self._generation_params = gen_params
        
        if USE_FAKE:
            # 纯伪装：直接返回期望答案
            rendered = self.tokenizer._last_rendered
            student_id = "S001"
            match = re.search(r"学生ID:\s*(\S+)", rendered)
            if match:
                student_id = match.group(1).strip()
            judgements = EXPECTED_ANSWERS.get(student_id, [True, True, True, True, True])
            result = {"student_id": student_id, "judgements": judgements}
            output_text = json.dumps(result, ensure_ascii=False)
        else:
            # 调用 Ollama
            messages = self.tokenizer._last_messages
            if not messages:
                # 兜底：从 rendered 文本粗略重建
                messages = [{"role": "user", "content": self.tokenizer._last_rendered}]
            output_text = call_ollama(messages, gen_params)
        
        self.tokenizer._last_output = output_text
        
        # 返回伪造 tensor
        input_len = getattr(input_ids, "shape", (1, 10))[1] if input_ids is not None else 10
        fake_output = torch.zeros((1, input_len + 20), dtype=torch.long)
        return fake_output


# ============================================================
# 5. Monkey Patch transformers
# ============================================================
def patch_transformers():
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel(fake_tokenizer)

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        print(f"[local_run] 使用伪装 Tokenizer（后端：{'纯伪装' if USE_FAKE else 'Ollama'}）")
        return fake_tokenizer

    def fake_model_from_pretrained(*args, **kwargs):
        mode = "纯伪装" if USE_FAKE else f"Ollama ({OLLAMA_MODEL})"
        print(f"[local_run] 使用伪装 Model（后端：{mode}）")
        return fake_model

    AutoTokenizer.from_pretrained = fake_tokenizer_from_pretrained
    AutoModelForCausalLM.from_pretrained = fake_model_from_pretrained
    transformers.AutoTokenizer.from_pretrained = fake_tokenizer_from_pretrained
    transformers.AutoModelForCausalLM.from_pretrained = fake_model_from_pretrained

    print("[local_run] transformers 已伪装完成")


# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print("LMCC T1 本地评测启动器（Ollama / 伪装模式）")
    print("=" * 60)
    print(f"Ollama Host : {OLLAMA_HOST}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print(f"模式        : {'纯伪装' if USE_FAKE else 'Ollama'}")
    print("=" * 60)

    patch_transformers()

    # 把当前目录和可能的 evaluate.py 所在目录加入 path
    current = Path(__file__).parent.resolve()
    if str(current) not in sys.path:
        sys.path.insert(0, str(current))

    try:
        import evaluate
    except ImportError:
        alt = current.parent / "attachments"
        if alt.exists():
            sys.path.insert(0, str(alt))
            import evaluate
        else:
            raise

    print("[local_run] 正在调用官方 evaluate.main() ...\n")

    # ============================================================
    # 把评测记录改存到 local_run.py 同级的 log/ 目录
    # ============================================================
    log_dir = Path(__file__).parent.resolve() / "log"
    log_dir.mkdir(exist_ok=True)
    record_file = log_dir / "evaluation_records.jsonl"

    def custom_save_evaluation_record(task_name: str, correct: int, total: int, score: int, total_time: float):
        try:
            record = {
                "timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task": task_name,
                "correct": correct,
                "total": total,
                "score": score,
                "total_time": round(total_time, 2)
            }
            with open(record_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"\n✅ 评测记录已保存到: {record_file}")
        except Exception as e:
            print(f"\n⚠ 保存评测记录失败: {e}")

    # 替换官方函数
    evaluate.save_evaluation_record = custom_save_evaluation_record
    print(f"[local_run] 评测记录将保存到: {record_file}")

    evaluate.main()


if __name__ == "__main__":
    main()
