import base64
import functools
import io
import json
import logging
import os
import platform
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, List, Sequence, TypedDict, cast

import chz
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

import tinker
from tinker import ModelInput
# 恢复直接导入 ensure_text
from tinker_cookbook.renderers import Message, Renderer, ensure_text, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
    Metrics,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.model_info import get_recommended_renderer_name

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: Constants & System Prompts
# ==============================================================================

SYSTEM_PROMPT_VISION = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will feature Numerical Labels placed in the TOP LEFT corner of each Web Element.
Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content. 
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage.
6. Google, directly jump to the Google search page. When you can't find information in some websites, try starting over with Google.
7. Answer. This action should only be chosen when all questions in the task have been solved.

Correspondingly, Action should STRICTLY follow the format:
- Click [Numerical_Label]
- Type [Numerical_Label]; [Content]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Wait
- GoBack
- Google
- ANSWER; [content]

Key Guidelines You MUST follow:
* Action guidelines *
1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
3) Execute only one action per iteration. 
4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
5) When a complex Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER. 
* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the numerical labels in the TOP LEFT corner of each rectangle (element). Ensure you don't mix them up with other numbers (e.g. Calendar) on the page.
4) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
5) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

Your reply should strictly follow the format:
Thought: {Your brief thoughts (briefly summarize the info that will help ANSWER)}
Action: {One Action format you choose}

Then the User will provide:
Observation: {A labeled screenshot Given by User}"""

SYSTEM_PROMPT_TEXT_ONLY = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Accessibility Tree with numerical label representing information about the page, then follow the guidelines and choose one of the following actions:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content. 
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is not the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage.
6. Google, directly jump to the Google search page. When you can't find information in some websites, try starting over with Google.
7. Answer. This action should only be chosen when all questions in the task have been solved.

Correspondingly, Action should STRICTLY follow the format:
- Click [Numerical_Label]
- Type [Numerical_Label]; [Content]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Wait
- GoBack
- Google
- ANSWER; [content]

Key Guidelines You MUST follow:
* Action guidelines *
1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
3) Execute only one action per iteration. 
4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
5) When a complex Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER. 
* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

Your reply should strictly follow the format:
Thought: {Your brief thoughts (briefly summarize the info that will help ANSWER)}
Action: {One Action format you choose}

Then the User will provide:
Observation: {Accessibility Tree of a web page}"""


# ==============================================================================
# SECTION 2: Selenium & WebArena Utils (Encapsulated)
# ==============================================================================

class WebController:
    """Encapsulates all Selenium and WebArena logic."""

    def __init__(self, headless: bool = True, window_size: tuple[int, int] = (1920, 1080), text_only: bool = False):
        self.headless = headless
        self.window_size = window_size
        self.text_only = text_only
        self.driver = None
        self._init_driver()

    def _init_driver(self):
        options = Options()
        options.page_load_strategy = 'normal'
        if self.headless:
            options.add_argument("--headless")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

        # Enable downloads
        options.add_experimental_option("prefs", {
            "plugins.always_open_pdf_externally": True
        })

        options.add_argument("--force-device-scale-factor=1.5")

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(*self.window_size)

    def navigate(self, url: str):
        try:
            self.driver.get(url)
            self._wait_for_stable_url()
            # Inject focus script
            try:
                self.driver.execute_script("window.focus();")
            except:
                pass
            # Inject scroll preventer
            self.driver.execute_script(
                """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Navigation failed: {e}")

    def _wait_for_stable_url(self, timeout=5.0):
        end_time = time.time() + timeout
        last_url = self.driver.current_url
        stable_counter = 0
        while time.time() < end_time:
            time.sleep(0.5)
            current_url = self.driver.current_url
            if current_url == last_url:
                stable_counter += 1
                if stable_counter >= 2: break
            else:
                last_url = current_url
                stable_counter = 0
                end_time = time.time() + timeout

    def get_web_element_rect(self, fix_color=True):
        selected_function = "getFixedColor" if fix_color else "getRandomColor"
        # JS Script from WebArena (Minified for brevity)
        js_script = """
            let labels = [];
            function markPage() {
                var bodyRect = document.body.getBoundingClientRect();
                var items = Array.prototype.slice.call(document.querySelectorAll('*')).map(function(element) {
                    var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
                    var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
                    var rects = [...element.getClientRects()].filter(bb => {
                        var center_x = bb.left + bb.width / 2;
                        var center_y = bb.top + bb.height / 2;
                        var elAtCenter = document.elementFromPoint(center_x, center_y);
                        return elAtCenter === element || element.contains(elAtCenter) 
                    }).map(bb => {
                        const rect = { left: Math.max(0, bb.left), top: Math.max(0, bb.top), right: Math.min(vw, bb.right), bottom: Math.min(vh, bb.bottom) };
                        return { ...rect, width: rect.right - rect.left, height: rect.bottom - rect.top }
                    });
                    var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);
                    return {
                        element: element,
                        include: (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT") ||
                                 (element.tagName === "BUTTON" || element.tagName === "A" || (element.onclick != null) || window.getComputedStyle(element).cursor == "pointer") ||
                                 (element.tagName === "IFRAME" || element.tagName === "VIDEO" || element.tagName === "LI" || element.tagName === "TD" || element.tagName === "OPTION"),
                        area, rects, text: element.textContent.trim().replace(/\\s{2,}/g, ' ')
                    };
                }).filter(item => item.include && (item.area >= 20));

                const buttons = Array.from(document.querySelectorAll('button, a, input[type="button"], div[role="button"]'));
                items = items.filter(x => !buttons.some(y => items.some(z => z.element === y) && y.contains(x.element) && !(x.element === y) ));
                items = items.filter(x => !(x.element.parentNode && x.element.parentNode.tagName === 'SPAN' && x.element.parentNode.children.length === 1 && x.element.parentNode.getAttribute('role') && items.some(y => y.element === x.element.parentNode)));
                items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)))

                function getRandomColor(index) { var letters = '0123456789ABCDEF'; var color = '#'; for (var i = 0; i < 6; i++) { color += letters[Math.floor(Math.random() * 16)]; } return color; }
                function getFixedColor(index) { return '#000000'; }

                items.forEach(function(item, index) {
                    item.rects.forEach((bbox) => {
                        newElement = document.createElement("div");
                        var borderColor = COLOR_FUNCTION(index);
                        newElement.style.outline = `2px dashed ${borderColor}`;
                        newElement.style.position = "fixed";
                        newElement.style.left = bbox.left + "px";
                        newElement.style.top = bbox.top + "px";
                        newElement.style.width = bbox.width + "px";
                        newElement.style.height = bbox.height + "px";
                        newElement.style.pointerEvents = "none";
                        newElement.style.boxSizing = "border-box";
                        newElement.style.zIndex = 2147483647;
                        var label = document.createElement("span");
                        label.textContent = index;
                        label.style.position = "absolute";
                        label.style.top = Math.max(-19, -bbox.top) + "px";
                        label.style.left = Math.min(Math.floor(bbox.width / 5), 2) + "px";
                        label.style.background = borderColor;
                        label.style.color = "white";
                        label.style.padding = "2px 4px";
                        label.style.fontSize = "12px";
                        label.style.borderRadius = "2px";
                        newElement.appendChild(label);
                        document.body.appendChild(newElement);
                        labels.push(newElement);
                    });
                })
                return [labels, items]
            }
            return markPage();""".replace("COLOR_FUNCTION", selected_function)

        rects, items_raw = self.driver.execute_script(js_script)

        format_ele_text = []
        for web_ele_id in range(len(items_raw)):
            label_text = items_raw[web_ele_id]['text']
            ele_tag_name = items_raw[web_ele_id]['element'].tag_name
            ele_type = items_raw[web_ele_id]['element'].get_attribute("type")
            ele_aria_label = items_raw[web_ele_id]['element'].get_attribute("aria-label")
            input_attr_types = ['text', 'search', 'password', 'email', 'tel']

            if not label_text:
                if (
                        ele_tag_name.lower() == 'input' and ele_type in input_attr_types) or ele_tag_name.lower() == 'textarea' or (
                        ele_tag_name.lower() == 'button' and ele_type in ['submit', 'button']):
                    content = ele_aria_label if ele_aria_label else label_text
                    format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{content}\";")
            elif label_text and len(label_text) < 200:
                if not ("<img" in label_text and "src=" in label_text):
                    desc = f"\"{label_text}\""
                    if ele_aria_label and (ele_aria_label != label_text): desc += f", \"{ele_aria_label}\""
                    prefix = f"<{ele_tag_name}> " if ele_tag_name in ["button", "input", "textarea"] else ""
                    format_ele_text.append(f"[{web_ele_id}]: {prefix}{desc};")

        format_ele_text = '\t'.join(format_ele_text)
        return rects, [web_ele['element'] for web_ele in items_raw], format_ele_text

    def get_capture(self):
        """Captures observation: screenshot (base64) and SOM/Text."""
        time.sleep(0.5)
        rects, web_eles, web_eles_text = None, None, None

        # Mark elements
        try:
            rects, web_eles, web_eles_text = self.get_web_element_rect(fix_color=True)
        except Exception as e:
            logger.error(f"Error marking page: {e}")
            web_eles_text = "Error capturing element text."
            web_eles = []

        # Capture Screenshot
        screenshot_b64 = self.driver.get_screenshot_as_base64()
        
        # Resize screenshot to 720p while maintaining aspect ratio
        screenshot_b64 = self._resize_image_to_720p(screenshot_b64)
        base64_string = screenshot_b64
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]

        # 解码 Base64 字符串为二进制数据
        image_data = base64.b64decode(base64_string)

        # 保存到文件（例如保存为 PNG 格式）
        with open(f"output_image{time.time()}.png", "wb") as f:
            f.write(image_data)

        # Clean up red boxes
        if rects:
            for rect_ele in rects:
                try:
                    self.driver.execute_script("arguments[0].remove()", rect_ele)
                except:
                    pass

        return {
            "web_eles": web_eles,
            "web_text": web_eles_text,
            "screenshot": screenshot_b64
        }
        
    def _resize_image_to_720p(self, image_b64: str) -> str:
        """Resize image to 720p while maintaining aspect ratio."""
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Target height is 720, calculate width to maintain aspect ratio
        target_height = 720
        aspect_ratio = image.width / image.height
        target_width = int(target_height * aspect_ratio)
        
        # Resize image
        resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Convert back to base64
        buffer = io.BytesIO()
        resized_image.save(buffer, format='PNG')
        resized_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return resized_b64

    def execute_raw_action(self, action_type: str, args: dict, context: dict):
        web_eles = context.get('web_eles')
        try:
            if action_type == 'click':
                idx = int(args['id'])
                if 0 <= idx < len(web_eles):
                    ele = web_eles[idx]
                    self.driver.execute_script("arguments[0].setAttribute('target', '_self')", ele)
                    ele.click()
                    self._wait_for_stable_url(3.0)
                    return "Clicked."
                else:
                    return "Error: Element ID out of range."

            elif action_type == 'type':
                idx = int(args['id'])
                content = args['content']
                if 0 <= idx < len(web_eles):
                    ele = web_eles[idx]
                    ele.clear()
                    # Type safely
                    actions = ActionChains(self.driver)
                    actions.click(ele).send_keys(content).pause(0.5).send_keys(Keys.ENTER).perform()
                    self._wait_for_stable_url(5.0)
                    return f"Typed '{content}'."
                else:
                    return "Error: Element ID out of range."

            elif action_type == 'scroll':
                target = args['target']
                direction = args['direction']
                scroll_amount = self.window_size[1] * 2 // 3
                if target == "WINDOW":
                    script = f"window.scrollBy(0, {scroll_amount});" if direction == 'down' else f"window.scrollBy(0, {-scroll_amount});"
                    self.driver.execute_script(script)
                else:
                    # Scroll specific element
                    idx = int(target)
                    if 0 <= idx < len(web_eles):
                        ele = web_eles[idx]
                        actions = ActionChains(self.driver)
                        actions.move_to_element(ele)
                        key = Keys.ARROW_DOWN if direction == 'down' else Keys.ARROW_UP
                        actions.key_down(Keys.ALT).send_keys(key).key_up(Keys.ALT).perform()
                time.sleep(1)
                return f"Scrolled {direction}."

            elif action_type == 'wait':
                time.sleep(5)
                return "Waited 5s."

            elif action_type == 'goback':
                self.driver.back()
                self._wait_for_stable_url()
                return "Went back."

            elif action_type == 'google':
                self.navigate("https://www.google.com")
                return "Navigated to Google."

        except Exception as e:
            return f"Action Failed: {str(e)}"

        return "Unknown Action."

    def close(self):
        if self.driver: self.driver.quit()


# ==============================================================================
# SECTION 3: Tinker Environment
# ==============================================================================

@dataclass
class BrowserTask:
    annotation_id: str
    action_uid: str
    goal: str
    start_url: str
    history: list[Message]


class BrowserEnv(Env):
    def __init__(
        self,
        task: BrowserTask,
        renderer: Renderer,
        text_only: bool = False,
        headless: bool = True
    ):
        self.task = task
        self.renderer = renderer
        self.text_only = text_only
        self.browser = WebController(headless=headless, text_only=text_only)

        self.steps = 0
        self.history: list[Message] = list(task.history) if task.history else []  # Initialize with task's history
        self.last_context = {}
        self.done = False

        # Initial Navigation - Load the mhtml file based on annotation_id and action_uid
        mhtml_path = f"file://D:\\Globus\\{task.annotation_id}\\processed\\snapshots\\{task.action_uid}_before.mhtml"
        self.browser.navigate(mhtml_path)

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _format_msg(self, pdf_obs=None, warn_obs=None) -> Message:
        """Uses the prompt-provided format_msg logic."""

        # Get current state from browser
        capture = self.browser.get_capture()
        self.last_context = capture  # Cache for action execution

        web_img_b64 = capture['screenshot']
        web_text = capture['web_text']

        init_msg = f"Task Goal: {self.task.goal}\n"

        # Note: We only add the initial message if there's no existing history
        if len(self.history) <= 2:  # Only system message and initial user message
            init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"

            if self.text_only:
                return {'role': 'user', 'content': init_msg}

            # --- 修改点：使用 'image' 键，而不是 'image_url' ---
            return {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': f"data:image/png;base64,{web_img_b64}"},
                    {'type': 'text', 'text': init_msg},
                ]
            }
        else:
            prefix = ""
            if pdf_obs: prefix += f"Observation: {pdf_obs} "
            if warn_obs: prefix += f"Observation: {warn_obs} "

            text_prompt = f"{prefix}Please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"

            if self.text_only:
                return {'role': 'user', 'content': text_prompt}

            # --- 修改点：使用 'image' 键，而不是 'image_url' ---
            return {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': f"data:image/png;base64,{web_img_b64}"},
                    {'type': 'text', 'text': text_prompt},
                ]
            }

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        self.steps = 1

        # Select System Prompt
        sys_content = SYSTEM_PROMPT_TEXT_ONLY if self.text_only else SYSTEM_PROMPT_VISION
        sys_msg: Message = {"role": "system", "content": sys_content}

        user_msg = self._format_msg()
        # Only append to history if it's not already initialized
        if not self.history:
            self.history = [sys_msg, user_msg]
        else:
            # If history already exists (has previous steps), just add the new user message
            self.history.append(user_msg)

        return self.renderer.build_generation_prompt(self.history), self.stop_condition

    def _parse_and_execute(self, action_text: str) -> tuple[str, bool]:
        """Parses the strict action format and executes via browser."""
        action_text = action_text.strip()

        # Regex Matching based on System Prompt
        # 1. ANSWER; [content]
        if action_text.startswith("ANSWER"):
            return f"Answered: {action_text}", True

        # 2. Click [Numerical_Label]
        click_match = re.match(r"Click \[?(\d+)\]?", action_text, re.IGNORECASE)
        if click_match:
            return self.browser.execute_raw_action('click', {'id': click_match.group(1)}, self.last_context), False

        # 3. Type [Numerical_Label]; [Content]
        type_match = re.match(r"Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?", action_text, re.IGNORECASE)
        if type_match:
            return self.browser.execute_raw_action('type', {'id': type_match.group(1), 'content': type_match.group(2)},
                                                   self.last_context), False

        # 4. Scroll [Numerical_Label or WINDOW]; [up or down]
        scroll_match = re.match(r"Scroll \[?(\d+|WINDOW)\]?[; ]+\[?(up|down)\]?", action_text, re.IGNORECASE)
        if scroll_match:
            return self.browser.execute_raw_action('scroll', {'target': scroll_match.group(1),
                                                              'direction': scroll_match.group(2)},
                                                   self.last_context), False

        # 5. Wait
        if "Wait" in action_text:
            return self.browser.execute_raw_action('wait', {}, self.last_context), False

        # 6. GoBack
        if "GoBack" in action_text:
            return self.browser.execute_raw_action('goback', {}, self.last_context), False

        # 7. Google
        if "Google" in action_text:
            return self.browser.execute_raw_action('google', {}, self.last_context), False

        # --- MODIFICATION: Return specific Format Error string ---
        return "Invalid Action Format. Please strictly follow the format: 'Click [id]', 'Type [id]; [content]', etc. Check your syntax and try again.", False

    async def step(self, action: Action) -> StepResult:
        self.steps += 1

        # 1. Parse Model Output
        (action_message, _) = self.renderer.parse_response(action)
        model_content = ensure_text(action_message["content"])

        # Log model thought/action
        logtree.log_text(f"Step {self.steps} Model Output: {model_content}")

        # Extract "Action: ..." part if model outputs Thought + Action
        action_line = model_content
        if "Action:" in model_content:
            parts = model_content.split("Action:")
            if len(parts) > 1:
                action_line = parts[1].strip().split('\n')[0]  # Take the first line after Action:

        self.history.append({"role": "assistant", "content": model_content})

        # 2. Execute
        feedback, done = self._parse_and_execute(action_line)
        logtree.log_text(f"Execution Result: {feedback}")

        # 3. Reward Calculation (Modified for Penalty)
        reward = 0.0
        if done and "Answered" in feedback:
            reward = 1.0
        elif "Invalid Action Format" in feedback:
            # --- MODIFICATION: Apply Penalty ---
            reward = -1.0

        # 4. Get Next Observation
        next_obs_msg = None
        if not done:
            # We treat feedback as 'warn_obs'. If it was a format error, 'feedback' contains the warning.
            next_obs_msg = self._format_msg(warn_obs=feedback)
            self.history.append(next_obs_msg)
            next_input = self.renderer.build_generation_prompt(self.history)
        else:
            next_input = ModelInput.empty()

        return StepResult(
            next_observation=next_input,
            next_stop_condition=self.stop_condition,
            episode_done=done,
            reward=reward,
            metrics={"success": float(reward > 0), "format_error": float(reward < 0)}
        )


# ==============================================================================
# SECTION 4: Dataset & Builders with Step-by-Step Training Support
# ==============================================================================

@dataclass(frozen=True)
class BrowserEnvGroupBuilder(EnvGroupBuilder):
    tasks: list[BrowserTask]
    renderer: Renderer
    text_only: bool
    headless: bool

    async def make_envs(self) -> Sequence[Env]:
        return [
            BrowserEnv(task, self.renderer, self.text_only, self.headless)
            for task in self.tasks
        ]
    
    async def compute_group_rewards(
        self, 
        trajectory_group: list[Trajectory], 
        env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """
        计算组中每条轨迹的奖励，然后只将奖励最高的最优轨迹的输入输出加入到数据缓冲区中，
        确保历史累积数据的质量和有效性。
        """
        # 计算每条轨迹的奖励
        rewards = []
        for trajectory in trajectory_group:
            # 从轨迹中获取奖励信息
            total_reward = sum([step.reward for step in trajectory.steps])
            metrics = trajectory.steps[-1].metrics if trajectory.steps else {}
            rewards.append((total_reward, metrics))
        
        # 找到奖励最高的轨迹索引
        if rewards:
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i][0])
            
            # 从奖励最高的轨迹中提取信息并更新数据缓冲区
            best_trajectory = trajectory_group[best_idx]
            best_env = env_group[best_idx] if best_idx < len(env_group) else None
            
            if best_env and isinstance(best_env, BrowserEnv):
                # 从最优轨迹中提取最后一步的信息
                if best_trajectory.steps:
                    # 为了更新数据缓冲区，我们需要访问环境的history
                    # 这里我们直接使用环境的history，因为轨迹可能不包含完整的历史信息
                    if len(best_env.history) >= 2:
                        assistant_output = best_env.history[-1]  # Last message is assistant's response
                        user_input = best_env.history[-2]        # Second to last is user's input (which includes the observation)
                        
                        # 假设我们能从action_uid中提取step_index，或者使用其他方式
                        # 这里使用一个简化的步骤索引
                        step_index = best_env.steps - 1
                        annotation_id = best_env.task.annotation_id
                        
                        # Add this input-output pair to the data buffer for the next step
                        data_buffer.add_step_result(annotation_id, step_index, user_input, assistant_output)
        
        # 返回所有轨迹的奖励，但只将最优轨迹的数据加入缓冲区
        return [(0.0, {}) for _ in range(len(trajectory_group))]


@dataclass(frozen=True)
class BrowserDataset(RLDataset):
    tasks: Sequence[BrowserTask]
    renderer: Renderer
    batch_size: int
    group_size: int
    text_only: bool = False
    headless: bool = True

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.tasks))
        batch = self.tasks[start:end]

        builders = []
        for task in batch:
            # 所有任务都包含在训练中，因为history的验证逻辑已经移除
            # 只有在compute_group_rewards中完成history后才会更新数据缓冲区
            builders.append(
                BrowserEnvGroupBuilder(
                    tasks=[task] * self.group_size,
                    renderer=self.renderer,
                    text_only=self.text_only,
                    headless=self.headless,
                    num_envs=self.group_size,
                    dataset_name="web_browser"
                )
            )

        return builders

    def __len__(self) -> int:
        return (len(self.tasks) + self.batch_size - 1) // self.batch_size


def load_mind2web_steps_from_annotation(annotation_id: str, action_uid: str) -> list[dict]:
    """
    Load the specific step from Mind2Web dataset based on annotation_id and action_uid.
    This function would access the mind2web dataset to get the correct action for this step.
    """
    # Import the mind2web module to access its functions
    import mind2web
    
    # Use the load_dataset function from mind2web module
    from datasets import load_dataset
    dataset = load_dataset("osunlp/Mind2Web", split="train")
    
    # Find the sample with the given annotation_id
    for sample in dataset:
        if sample['annotation_id'] == annotation_id:
            # Find the specific action with the given action_uid
            for action in sample['actions']:
                if action['action_uid'] == action_uid:
                    return action  # Return the specific action data
    
    # If not found, return None or raise an exception
    return None


class Mind2WebDataBuffer:
    """
    Data buffer to maintain step-by-step training data from Mind2Web dataset.
    Each step's history includes the actual model input and output from previous steps.
    """
    def __init__(self):
        self.buffer = {}  # Maps (annotation_id, step_index) to task with accumulated history

    def add_step_result(self, annotation_id: str, step_index: int, model_input: Message, model_output: Message):
        """
        Add model input and output to the history for the next step.
        """
        key = (annotation_id, step_index + 1)  # Next step gets the history
        if key not in self.buffer:
            self.buffer[key] = []
        
        # Add the input and output to the history for the next step
        self.buffer[key].extend([model_input, model_output])
    
    def get_history_for_step(self, annotation_id: str, step_index: int) -> list[Message]:
        """
        Get accumulated history for a specific step.
        """
        key = (annotation_id, step_index)
        return self.buffer.get(key, [])


# Global data buffer instance to maintain history across rollouts
data_buffer = Mind2WebDataBuffer()


def create_mind2web_tasks_with_dynamic_context() -> list[BrowserTask]:
    """
    Create BrowserTask objects without binding to specific tasks initially.
    Context is fetched dynamically when needed during execution.
    """
    import mind2web
    from datasets import load_dataset
    
    # Load the Mind2Web dataset
    dataset = load_dataset("osunlp/Mind2Web", split="train")
    
    tasks = []
    
    # Limit the number of samples for testing
    sample_count = 0
    max_samples = 3  # Limit for testing purposes
    
    for sample in dataset:
        if sample_count >= max_samples:
            break
            
        # Check if the required directory exists
        if not os.path.exists(f"D:\\Globus\\{sample['annotation_id']}\\processed\\snapshots"):
            continue
        
        # Create tasks for each action in the sample with empty history initially
        # Context will be fetched dynamically during execution
        for step_idx, action in enumerate(sample['actions']):
            # For the first step, initialize with empty history
            # For subsequent steps, the history will be populated dynamically when available
            task = BrowserTask(
                annotation_id=sample['annotation_id'],
                action_uid=action['action_uid'],
                goal=sample['confirmed_task'],
                start_url=sample.get('start_url', ''),
                history=[]  # Start with empty history for all steps initially
            )
            
            tasks.append(task)
        
        sample_count += 1
        
    return tasks


@chz.chz
class BrowserDatasetBuilder(RLDatasetBuilder):
    """
    Builder for Browser RL Tasks with step-by-step training support.
    """
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int

    # Custom Configs
    text_only: bool = False
    headless: bool = True

    def _generate_tasks_with_context(self) -> list[BrowserTask]:
        """Generates tasks with context fetched dynamically during execution."""
        return create_mind2web_tasks_with_dynamic_context()

    def _generate_dummy_tasks(self) -> list[BrowserTask]:
        """Generates placeholder tasks as requested."""
        return [
            BrowserTask("1", "action_1", "Find the price of iPhone 15 on Amazon", "https://www.amazon.com", []),
            BrowserTask("2", "action_2", "Search for 'Tinker RL' on Google", "https://www.google.com", []),
            BrowserTask("3", "action_3", "Find the latest news on CNN", "https://www.cnn.com", []),
        ] * 10  # Repeat to simulate a larger dataset

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Use the Mind2Web dataset to generate tasks
        tasks = self._generate_tasks_with_context()

        # Split 80/20
        split_idx = int(len(tasks) * 0.8)
        train_tasks = tasks[:split_idx]
        test_tasks = tasks[split_idx:]

        train_ds = BrowserDataset(
            tasks=train_tasks,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            text_only=self.text_only,
            headless=self.headless
        )

        test_ds = BrowserDataset(
            tasks=test_tasks,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=1,  # Test usually uses group_size=1
            text_only=self.text_only,
            headless=self.headless
        )

        return train_ds, test_ds


# Function to update the data buffer after a rollout step is completed
def update_data_buffer_after_rollout(annotation_id: str, step_index: int, env: BrowserEnv):
    """
    Updates the data buffer with the actual model input and output after a rollout step.
    This function should be called after each rollout step to maintain the history.
    
    Args:
        annotation_id: The annotation ID of the current task
        step_index: The index of the current step
        env: The environment that just completed the step
    """
    # Get the last input (user message) and output (assistant message) from the environment history
    if len(env.history) >= 2:
        # The last two messages should be the most recent input and output
        # In the history, the last message is the assistant's output, and the one before is the user's input
        assistant_output = env.history[-1]  # Last message is assistant's response
        user_input = env.history[-2]        # Second to last is user's input (which includes the observation)
        
        # Add this input-output pair to the data buffer for the next step
        data_buffer.add_step_result(annotation_id, step_index, user_input, assistant_output)