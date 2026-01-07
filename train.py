"""
CLI Skeleton for Generic RL Training
"""

import asyncio
from datetime import datetime
import chz
from tinker_cookbook.rl import train
import os

# 假设这里有一个通用的或自定义的 DatasetBuilder
from env import BrowserPool, BrowserDatasetBuilder

from tinker_cookbook.model_info import get_recommended_renderer_name

os.environ['TINKER_API_KEY'] = 'tml-Wrcd7jkyejehmtjAfQ8uUgyfyWtOwWQX8GCIqI6esrtLfD0FxsT6AiISJ5OPGovmjAAAA'

@chz.chz
class CLIConfig:
    # --- 模型参数 ---
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32

    # --- 训练参数 ---
    learning_rate: float = 4e-5
    batch_size: int = 4
    seed: int = 42
    max_tokens: int = 512
    eval_every: int = 0

    # --- 性能优化配置 (Streaming) ---
    stream_minibatch: bool = False
    num_minibatches: int = 4

    # --- 日志参数 ---
    log_path: str | None = "log"
    wandb_project: str | None = None
    wandb_name: str | None = None


async def cli_main(cli_config: CLIConfig):
    # 1. 配置 Streaming Minibatch (通用训练策略)
    stream_minibatch_config = None
    if cli_config.stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size,
            num_minibatches=cli_config.num_minibatches,
        )

    # 2. 构建 Dataset Builder (此处需要根据具体任务实现)
    # 示例: builder = CustomDatasetBuilder(batch_size=cli_config.batch_size, seed=cli_config.seed)
    pool = BrowserPool(headless=False)
    builder = BrowserDatasetBuilder(batch_size=cli_config.batch_size,
                                    model_name_for_tokenizer=cli_config.model_name,
                                    renderer_name=get_recommended_renderer_name(cli_config.model_name),
                                    group_size=4,
                                    pool=pool)

    # 3. 生成 Run Name (通用格式)
    if cli_config.wandb_name:
        wandb_name = cli_config.wandb_name
    else:
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
        wandb_name = f"run_{cli_config.model_name.split('/')[-1]}_{date_str}"

    # 4. 构建核心训练配置
    config = train.Config(
        model_name=cli_config.model_name,
        log_path=cli_config.log_path,  # 如果为None，底层通常会处理默认路径
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        stream_minibatch_config=stream_minibatch_config,
    )

    # 5. 执行训练
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))