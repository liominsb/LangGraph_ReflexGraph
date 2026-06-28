import os
import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# ── 环境变量 ──
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")

# ── 全局常量 ──
MAX_TOKENS = 60000
TIMEOUT = httpx.Timeout(100.0)

# ── 主力模型（强推理） ──
llm = init_chat_model(
    "mimo-v2.5-pro",
    model_provider="openai",
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=TIMEOUT,
    max_retries=10,
)

# ── 轻量模型（摘要/压缩） ──
llm_flash = init_chat_model(
    "mimo-v2.5",
    model_provider="openai",
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=TIMEOUT,
    max_retries=10,
)
