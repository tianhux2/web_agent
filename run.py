import asyncio
import logging
import argparse
import os

# Tinker 核心库
import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.model_info import get_recommended_renderer_name

# --- 修改点：导入 image_processing_utils ---
from tinker_cookbook.image_processing_utils import get_image_processor

# 导入你之前定义的 Environment 代码
from env import BrowserEnv, BrowserTask

# 请替换为你的 API Key
os.environ['TINKER_API_KEY'] = 'tml-Wrcd7jkyejehmtjAfQ8uUgyfyWtOwWQX8GCIqI6esrtLfD0FxsT6AiISJ5OPGovmjAAAA'

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_single_episode(args):
    # =========================================================================
    # 修改点: 极简初始化
    # =========================================================================

    # 1. 初始化 ServiceClient
    service_client = tinker.ServiceClient()

    logger.info(f"Connected to Tinker Service. Model: {args.model_name}")

    # 2. 创建采样客户端
    sampling_client = service_client.create_sampling_client(
        base_model=args.model_name
    )

    # =========================================================================
    # Renderer 初始化优化
    # =========================================================================

    # 获取本地 Tokenizer
    tokenizer = get_tokenizer(args.model_name)

    # --- 修改点：使用 tinker_cookbook 的工具获取 image processor ---
    logger.info("Loading Image Processor...")
    try:
        image_processor = get_image_processor(args.model_name)
    except Exception as e:
        logger.error(f"Failed to load image processor: {e}")
        raise e

    renderer_name = args.renderer_name or get_recommended_renderer_name(args.model_name)

    logger.info(f"Initializing Renderer: {renderer_name}")
    # --- 修改点：传入 image_processor ---
    renderer = get_renderer(
        renderer_name,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    # 初始化 Agent
    agent_completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=128,
    )

    # 3. 初始化环境
    task = BrowserTask(
        id="test_cloud",
        goal=args.goal,
        start_url=args.url
    )

    # headless=False: 本地会弹出浏览器窗口，你可以看着 AI 操作
    env = BrowserEnv(task, renderer, text_only=args.text_only, headless=False)

    print(f"\n🚀 Starting Task: {task.goal}")
    print(f"🌐 URL: {task.start_url}")
    print("-" * 50)

    try:
        # 获取初始页面
        obs, stop_condition = await env.initial_observation()

        done = False
        step_count = 0
        max_steps = 15

        while not done and step_count < max_steps:
            step_count += 1
            print(f"\n[Step {step_count}] Requesting Remote Inference...")

            print(type(obs))
            # 发送截图和文本到 Tinker 云端，等待返回 Action
            completion = await agent_completer(obs)

            model_output = completion["content"]
            print(f"🤖 Model Action: {model_output}")

            # 本地浏览器执行 Action
            step_result = await env.step(completion)

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
        env.browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 你的模型名称，例如 "Qwen/Qwen3-VL-30B-A3B-Instruct"
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="Tinker 平台上的 Base Model ID")
    parser.add_argument("--renderer_name", type=str, default=None)
    parser.add_argument("--goal", type=str, default="Search for 'Tinker RL' on Google", help="任务目标")
    parser.add_argument("--url", type=str, default="https://www.google.com", help="起始URL")
    parser.add_argument("--text_only", action="store_true")

    args = parser.parse_args()

    asyncio.run(run_single_episode(args))