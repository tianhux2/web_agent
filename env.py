import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Sequence

import chz
from tinker import ModelInput

from browser_env import BrowserPool, WebController, SYSTEM_PROMPT_VISION
from reward import calculate_reward
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.image_processing_utils import get_image_processor
# 恢复直接导入 ensure_text
from tinker_cookbook.renderers import Message, Renderer, ensure_text, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 3: Tinker Environment (Updated)
# ==============================================================================

@dataclass
class BrowserTask:
    id: str
    goal: str
    start_url: str

class BrowserEnv(Env):
    def __init__(
        self,
        task: BrowserTask,
        renderer: Renderer,
        pool: BrowserPool,  # Pass Pool instead of creating browser
    ):
        self.task = task
        self.renderer = renderer
        self.pool = pool

        self.browser: WebController | None = None  # Initialized in setup
        self.steps = 0
        self.history: list[Message] = []
        self.last_context = {}
        self.done = False

    async def setup(self):
        """Manually called to acquire resources."""
        # Asynchronously wait for a browser from the pool
        self.browser = await self.pool.acquire()

        # Initial Navigation (in thread)
        await asyncio.to_thread(self.browser.navigate, self.task.start_url)

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def _get_obs_async(self):
        """Wrapper to get observation in thread."""
        return await asyncio.to_thread(self.browser.get_capture)

    async def _format_msg(self, warn_obs=None) -> Message:
        """Uses the prompt-provided format_msg logic (ASYNC version)."""

        # Get current state from browser (ASYNC)
        capture = await self._get_obs_async()
        self.last_context = capture

        web_img_b64 = capture['screenshot']
        web_text = capture['web_text']

        init_msg = f"Task Goal: {self.task.goal}\n"

        if self.steps == 1:
            init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"

            return {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': f"data:image/png;base64,{web_img_b64}"},
                    {'type': 'text', 'text': init_msg},
                ]
            }
        else:
            prefix = ""
            if warn_obs: prefix += f"Observation: {warn_obs} "

            text_prompt = f"{prefix}Please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"

            return {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': f"data:image/png;base64,{web_img_b64}"},
                    {'type': 'text', 'text': text_prompt},
                ]
            }

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        self.steps = 1

        # Ensure setup is called if not already
        if not self.browser:
            await self.setup()

        # Select System Prompt
        sys_content = SYSTEM_PROMPT_VISION
        sys_msg: Message = {"role": "system", "content": sys_content}

        user_msg = await self._format_msg()  # Now awaited
        self.history = [sys_msg, user_msg]

        return self.renderer.build_generation_prompt(self.history), self.stop_condition

    def _parse_and_execute_sync(self, action_text: str) -> tuple[str, bool]:
        """Strict logic for parsing and executing (Blocking, runs in thread)."""
        action_text = action_text.strip()

        if action_text.startswith("ANSWER"):
            return f"Answered: {action_text}", True

        click_match = re.match(r"Click \[?(\d+)]?", action_text, re.IGNORECASE)
        if click_match:
            return self.browser.execute_raw_action('click', {'id': click_match.group(1)}, self.last_context), False

        type_match = re.match(r"Type \[?(\d+)]?[; ]+\[?(.[^]]*)]?", action_text, re.IGNORECASE)
        if type_match:
            return self.browser.execute_raw_action('type', {'id': type_match.group(1), 'content': type_match.group(2)},
                                                   self.last_context), False

        scroll_match = re.match(r"Scroll \[?(\d+|WINDOW)]?[; ]+\[?(up|down)]?", action_text, re.IGNORECASE)
        if scroll_match:
            return self.browser.execute_raw_action('scroll', {'target': scroll_match.group(1),
                                                              'direction': scroll_match.group(2)},
                                                   self.last_context), False

        if "Wait" in action_text:
            return self.browser.execute_raw_action('wait', {}, self.last_context), False

        if "GoBack" in action_text:
            return self.browser.execute_raw_action('goback', {}, self.last_context), False

        if "Google" in action_text:
            return self.browser.execute_raw_action('google', {}, self.last_context), False

        return "Invalid Action Format.", False

    async def step(self, action: Action) -> StepResult:
        self.steps += 1

        # 1. Parse Model Output
        (action_message, _) = self.renderer.parse_response(action)
        model_content = ensure_text(action_message["content"])

        logtree.log_text(f"Step {self.steps} Model Output: {model_content}")
        print(f"Step {self.steps} Model Output: {model_content}")

        action_line = model_content
        if "Action:" in model_content:
            parts = model_content.split("Action:")
            if len(parts) > 1:
                action_line = parts[1].strip().split('\n')[0]

        self.history.append({"role": "assistant", "content": model_content})

        # 2. Execute (ASYNC WRAPPER)
        # We run the synchronous parse_and_execute logic in a thread
        feedback, done = await asyncio.to_thread(self._parse_and_execute_sync, action_line)

        logtree.log_text(f"Execution Result: {feedback}")
        print(f"Execution Result: {feedback}; Done: {done}")

        # 3. Reward Calculation
        reward = 0.0
        if done and "Answered" in feedback:
            reward = calculate_reward(self.history)
        elif "Invalid Action Format" in feedback:
            reward = -1.0

        print(f"Reward: {reward}")

        # 4. Get Next Observation
        if not done:
            next_obs_msg = await self._format_msg(warn_obs=feedback)  # Now awaited
            self.history.append(next_obs_msg)
            next_input = self.renderer.build_generation_prompt(self.history)
        else:
            next_input = ModelInput.empty()
            await self.close()

        return StepResult(
            next_observation=next_input,
            next_stop_condition=self.stop_condition,
            episode_done=done,
            reward=reward,
            metrics={"success": float(reward > 0), "format_error": float(reward < 0)}
        )

    async def close(self):
        """Release the browser back to the pool."""
        if self.browser:
            await self.pool.release(self.browser)
            self.browser = None


# ==============================================================================
# SECTION 4: Dataset & Builders
# ==============================================================================

@dataclass(frozen=True)
class BrowserEnvGroupBuilder(EnvGroupBuilder):
    tasks: list[BrowserTask]
    renderer: Renderer
    pool: BrowserPool

    async def make_envs(self) -> Sequence[Env]:
        print("tasks:", [(task.goal, task.start_url) for task in self.tasks])
        return [
            BrowserEnv(task, self.renderer, self.pool)
            for task in self.tasks
        ]


@dataclass(frozen=True)
class BrowserDataset(RLDataset):
    tasks: Sequence[BrowserTask]
    renderer: Renderer
    batch_size: int
    group_size: int
    pool: BrowserPool

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.tasks))
        batch = self.tasks[start:end]

        builders = []
        for task in batch:
            builders.append(
                BrowserEnvGroupBuilder(
                    tasks=[task] * self.group_size,
                    renderer=self.renderer,
                    pool=self.pool,
                )
            )
        return builders

    def __len__(self) -> int:
        return (len(self.tasks) + self.batch_size - 1) // self.batch_size


@chz.chz
class BrowserDatasetBuilder(RLDatasetBuilder):
    """
    Builder for Browser RL Tasks.
    """
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    pool: BrowserPool



    def _generate_dummy_tasks(self) -> list[BrowserTask]:
        """Generates placeholder tasks as requested."""
        return [
            BrowserTask("1", "Translate hello world to Chinese", "https://www.iciba.com/"),
            BrowserTask("2", "Translate hello world to Chinese", "https://translate.google.com/"),
        ] * 2  # Repeat to simulate a larger dataset

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

        tasks = self._generate_dummy_tasks()

        # Split 80/20
        split_idx = int(len(tasks) * 0.8)
        train_tasks = tasks[:split_idx]
        test_tasks = tasks[split_idx:]

        train_ds = BrowserDataset(
            tasks=train_tasks,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            pool=self.pool
        )

        test_ds = BrowserDataset(
            tasks=test_tasks,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=1,  # Test usually uses group_size=1
            pool=self.pool
        )

        return train_ds, test_ds