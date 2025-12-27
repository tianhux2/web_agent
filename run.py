import asyncio
import logging
import argparse
import os

# 导入 Tinker 核心库
import tinker
from tinker_cookbook.utils import logtree
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.model_info import get_recommended_renderer_name

# 导入你之前定义的 Environment 代码
# 假设你把之前的代码保存为 browser_env_def.py
from browser_env_def import BrowserEnv, BrowserTask, SYSTEM_PROMPT_VISION

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_single_episode(args):
    # 1. 配置模型连接 (Agent)
    # 使用 Tinker 的 ServiceClient 连接到本地的 vLLM 服务
    service_client = tinker.ServiceClient(base_url=args.base_url, api_key="EMPTY")

    # 创建采样客户端
    sampling_client = service_client.create_sampling_client(base_model=args.model_name)

    # 获取 Tokenizer 和 Renderer (用于处理 Prompt template)
    tokenizer = get_tokenizer(args.model_name)
    # 如果你的模型是 Llama3 且带有 Vision，可能需要特定的 renderer，这里使用推荐的
    renderer_name = args.renderer_name or get_recommended_renderer_name(args.model_name)
    renderer = get_renderer(renderer_name, tokenizer=tokenizer)

    # 初始化 Completer (它负责将 Obs 发送给模型并获取回复)
    agent_completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=128,  # 控制输出长度，动作通常很短
        temperature=0.0,  # 测试时通常使用 0 温度以获得确定性结果
    )

    # 2. 初始化环境 (Environment)
    task = BrowserTask(
        id="test_001",
        goal=args.goal,  # 例如: "Find the price of iPhone 15 on Amazon"
        start_url=args.url  # 例如: "https://www.amazon.com"
    )

    # headless=False 可以在本地弹窗看到浏览器自动操作的效果
    env = BrowserEnv(task, renderer, text_only=args.text_only, headless=False)

    print(f"\n🚀 Starting Task: {task.goal}")
    print(f"🌐 URL: {task.start_url}")
    print("-" * 50)

    try:
        # 3. 获取初始观察 (Observation)
        # obs 包含了 Prompt (System Prompt + 截图 + DOM Tree)
        obs, stop_condition = await env.initial_observation()

        done = False
        step_count = 0
        max_steps = 15

        while not done and step_count < max_steps:
            step_count += 1
            print(f"\n[Step {step_count}] Thinking...")

            # 4. Agent 推理 (Model Inference)
            # 将环境的观察 (obs) 发送给模型，并传入停止词 (stop_condition)
            # completion 是模型生成的文本 (例如: "Action: Click [15]")
            completion = await agent_completer(obs, stop_sequences=stop_condition)

            model_output = completion["content"]
            print(f"🤖 Model Action: {model_output}")

            # 5. 环境执行 (Environment Step)
            # 将模型的输出传回环境，环境解析动作、执行 Selenium 操作、计算奖励
            step_result = await env.step(completion)

            # 6. 更新状态
            obs = step_result.next_observation
            done = step_result.episode_done
            reward = step_result.reward

            if done:
                print("-" * 50)
                status = "SUCCESS" if reward > 0 else "FAILED"
                print(f"🏁 Episode Finished. Result: {status} (Reward: {reward})")

        if not done:
            print(f"❌ Timed out after {max_steps} steps.")

    finally:
        # 关闭浏览器
        env.browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--model_name", type=str, required=True, help="vLLM启动的服务模型名称")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="推理服务的地址")
    parser.add_argument("--renderer_name", type=str, default=None, help="Tinker renderer 名称 (如 llama3, qwen2)")

    # 任务参数
    parser.add_argument("--goal", type=str, default="Search for 'Tinker RL' on Google", help="任务目标")
    parser.add_argument("--url", type=str, default="https://www.google.com", help="起始URL")

    # 环境配置
    parser.add_argument("--text_only", action="store_true", help="是否仅使用纯文本模式 (无截图)")

    args = parser.parse_args()

    asyncio.run(run_single_episode(args))