import asyncio
import logging
import argparse
import os
import time

# Tinker 核心库
import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.model_info import get_recommended_renderer_name
# Image Processor
from tinker_cookbook.image_processing_utils import get_image_processor

# 导入 Environment
from env import BrowserEnv, BrowserTask, BrowserPool

# API Key
os.environ['TINKER_API_KEY'] = 'tml-Wrcd7jkyejehmtjAfQ8uUgyfyWtOwWQX8GCIqI6esrtLfD0FxsT6AiISJ5OPGovmjAAAA'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_single_episode(args):
    # 1. 初始化 Service
    service_client = tinker.ServiceClient()
    logger.info(f"Connected to Tinker Service. Model: {args.model_name}")

    sampling_client = service_client.create_sampling_client(
        base_model=args.model_name
    )

    # 2. 本地组件
    tokenizer = get_tokenizer(args.model_name)
    logger.info("Loading Image Processor...")
    image_processor = get_image_processor(args.model_name)

    renderer_name = args.renderer_name or get_recommended_renderer_name(args.model_name)
    logger.info(f"Initializing Renderer: {renderer_name}")

    renderer = get_renderer(
        renderer_name,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    # 3. 环境初始化
    task = BrowserTask(
        id="test_cloud",
        goal=args.goal,
        start_url=args.url
    )
    pool = BrowserPool(headless=False)
    env = BrowserEnv(task, renderer, text_only=args.text_only, pool=pool)

    print(f"\n🚀 Starting Task: {task.goal}")
    print(f"🌐 URL: {task.start_url}")
    print("-" * 50)

    try:
        # 获取 ModelInput
        obs, stop_condition = await env.initial_observation()

        done = False
        step_count = 0
        max_steps = 3

        while not done and step_count < max_steps:
            step_count += 1
            print(f"\n[Step {step_count}] Requesting Remote Inference...")

            # =================================================================
            # 4. 模型推理 (Fix: 参数配置与结果提取)
            # =================================================================

            # 构造 SamplingParams
            # 注意：根据 Tinker 版本不同，参数名可能是 stop 或 stop_sequences
            # 只要之前没报参数错误，说明 stop_sequences 是对的
            params = tinker.SamplingParams(
                max_tokens=128,
                temperature=0.0,
                stop_sequences=stop_condition
            )

            start_time = time.time()
            # 调用 sample()
            future = sampling_client.sample(
                prompt=obs,
                sampling_params=params,
                num_samples=1
            )

            # 获取结果
            result = future.result()
            print(f"time used: {time.time() - start_time}")

            # --- 关键修复：从 SampleResponse 中提取 Token 序列 ---
            # result 是 SampleResponse 对象
            # result.sequences 是一个列表，我们取第一个采样结果
            # sequence.tokens 才是真正的 Action (List[int])
            if not result.sequences:
                logger.error("No sequences returned from model!")
                break

            # 提取动作 (Token IDs)
            action = result.sequences[0].tokens

            # =================================================================
            # 5. 解析与执行
            # =================================================================

            # 解析文本用于打印 (传入 Action 列表，而不是 Response 对象)
            (message, _) = renderer.parse_response(action)

            raw_content = message["content"]
            if isinstance(raw_content, list):
                model_output = "".join([x.get("text", "") for x in raw_content if x.get("type") == "text"])
            else:
                model_output = str(raw_content)

            print(f"🤖 Model Action: {model_output}")

            # 5. 环境执行 (传入 Action 列表)
            step_result = await env.step(action)

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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="Tinker 平台上的 Base Model ID")
    parser.add_argument("--renderer_name", type=str, default=None)
    parser.add_argument("--goal", type=str, default="Translate hello world to Chinese", help="任务目标")
    parser.add_argument("--url", type=str, default="https://www.iciba.com/", help="起始URL")
    parser.add_argument("--text_only", action="store_true")

    args = parser.parse_args()

    asyncio.run(run_single_episode(args))