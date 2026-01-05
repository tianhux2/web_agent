import os
import time
from datasets import load_dataset
from selenium import webdriver
from pathlib import Path


def display_mind2web_steps(sample_limit=1):
    # 1. 加载数据集 (使用流式加载以节省内存)
    dataset = load_dataset("osunlp/Mind2Web", split="train")

    # 2. 初始化浏览器 (以 Chrome 为例)
    driver = webdriver.Chrome()

    try:
        for i, sample in enumerate(dataset):
            if not Path(f"D:\\Globus\\{sample['annotation_id']}\\processed\\snapshots").is_dir():
                continue

            print(f"--- 正在展示任务 {i + 1}: {sample['confirmed_task']} ---")
            print(f"网站: {sample['website']}")
            print(f"任务: {sample['confirmed_task']}")
            print(sample['action_reprs'])

            jump = False
            for step_idx, action in enumerate(sample['actions']):
                if action['operation']['op'] == 'SELECT':
                    jump = True
                    break

            # if jump:
            #     continue

            # 遍历任务中的每一个动作 (Action)
            for step_idx, action in enumerate(sample['actions']):

                # 4. Selenium 加载本地文件
                driver.get(f"file://D:\\Globus\\{sample['annotation_id']}\\processed\\snapshots\\{action['action_uid']}_before.mhtml")

                # 打印当前步骤的操作信息
                op = action['operation']
                print(f"步骤 {step_idx + 1}: 操作类型 [{op['op']}], 值: '{op['value']}'")

                print(action.keys())
                print(action['neg_candidates'])
                print(action['pos_candidates'])

                # 提示用户
                input("按回车键查看下一步...")
    finally:
        driver.quit()
        print("演示结束，浏览器已关闭。")


if __name__ == "__main__":
    # 运行演示
    display_mind2web_steps(sample_limit=2)