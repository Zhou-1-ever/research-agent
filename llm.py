import os

from dotenv import load_dotenv
from openai import OpenAI


# 读取 .env
load_dotenv()

# 获取配置
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")


if not api_key:
    raise RuntimeError("未读到 DEEPSEEK_API_KEY，请检查 .env")

# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
