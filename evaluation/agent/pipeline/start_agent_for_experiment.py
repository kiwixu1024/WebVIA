import requests
import base64
import time
import json
import subprocess
import psutil
import os
import hashlib
import sys
from tqdm import tqdm
from multiprocessing import Process, Queue
import re
import copy
import ast
import shutil
import openai
from openai import OpenAI
import argparse

import json
import os
import argparse
import base64
import requests
import time
import shutil
import psutil
import subprocess
import sys
import re
import copy
import ast
import hashlib
from tqdm import tqdm
from multiprocessing import Process, Queue
import openai
from openai import OpenAI

# ===== 新增：统一总目录 =====
ROOT_OUTPUT_DIR = "ALL_model_result"

# ===== 从配置文件加载 OpenAI 参数 =====
def load_api_config(config_path="../api_config.json"):
    """从 JSON 文件加载 api_key 和 api_base"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "api_key" not in cfg or "api_base" not in cfg:
        raise ValueError("配置文件必须包含 'api_key' 和 'api_base' 字段")
    return cfg["api_base"], cfg["api_key"]

# ===== 加载配置 =====
api_base, api_key = load_api_config("../api_config.json")

openai.api_base = api_base
openai.api_key = api_key

client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)

# ===== 模型清单 =====
MODELS = [
    "o4-mini-2025-04-16",
    "claude-sonnet-4-20250514-thinking",
    "gpt-5-2025-08-07",
    "gemini-2.5-pro",
    "claude-3-7-sonnet-20250219-thinking",
    "gpt-4o-2024-11-20",
]

input_dir = './htmls'

# # 保持原有全局名，循环时会覆盖它

def get_response(message):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=message,
        max_completion_tokens=10000,
        timeout=600
    )
    print(response)
    return response

def cp_bad_html(src_file):
    print("正在复制html")
    basename = os.path.basename(src_file)
    copypath = os.path.join(BUG_DIR, basename)
    shutil.copy(src_file, copypath)

def extract_boxed_text(text):
    match = re.search(r'\\+boxed\{([^}]*)\}', text)
    if match:
        return match.group(1)
    return ""

def find_node_by_id_for_prompt(domtree, node_id):
    if not domtree:
        return None
    if isinstance(domtree, str):
        try:
            domtree = ast.literal_eval(domtree)
        except Exception:
            print(f"[WARN] domtree字符串转dict失败，domtree:{domtree}")
            return None
    if str(domtree.get('id')) == str(node_id):
        return {
            "id": domtree.get("id"),
            "tag": domtree.get("tag"),
            "attrs": domtree.get("attrs"),
            "visible_text": domtree.get("visible_text"),
        }
    for child in domtree.get('children', []):
        res = find_node_by_id_for_prompt(child, node_id)
        if res:
            return res
    return None

def check_yes_no(boxed_text):
    if boxed_text == "有":
        return True
    elif boxed_text == "无":
        return False
    else:
        return None

def get_compare_response(img_b64_list, interact_name, compare_messages):
    prompt_base = (
        "你会收到多张网页图片作为互动历史的验证过程，"
        "多张网页图片呈正序排列，最后一张图片代表互动完成，第一张图片是起始图，每进行一次操作会截一次图，直到最后一张完成操作。比如如果只有两张图，就是只进行了一个操作，操作前截图是第一张，操作后截图是第二张。如果有四张图，就是进行了三次操作，操作前截图是第一张，第一次操作后是第二张，第二次操作是第三章，以此类推。"
        "请完成以下两个内容：\n"
        "1. 判断本次互动序列后网页是否产生了符合互动组件的变化，比如点击编辑会弹出编辑窗口，在编辑窗口输入一系列内容后点击“保存”会讲刚才输入的内容完整地保存到页面上，点击“取消”或者“关闭”后不会保存。我会给你这次点击的互动组件序列名字叫什么，有哪些内容。请认真观察网页，理解网页进行这一轮操作之后应该会出现哪些变化，重点判断在最后一次操作之后图片是否发生了这一系列动作应该有的变化。没有发生的变化比如在点击一个按钮后只有按钮本身高亮了，网页本身无变化。或者点击保存后退出了弹窗，但网页中没有真的保存。给出一个判断，回复我“无”表示两张图片大体上无变化，“有”表示发生了变化。请注意。将你给出的答案提取出来放在latex的\\boxed{}中，并且给出你判断的理由"
        "2. 请你仔细比较本次序列生成的最后一张图片，判断网页与本次的起始图相比是否出现了新的重要部分。如果新部分出现，需要继续检测（即页面与所有历史图片都不极度相似），请用\\terminate{}包裹“继续”；"
        "请注意！如果没有新变化（如页面整体无明显新部分/出现内容与之前某张历史图极度相似/页面中出现的新部分极其微小或互动内容过少不值得继续/页面中出现的新内容与原图中内容高度重复），例如在页面中删除了一个栏目，没有任何新的互动组件出现。请用\\terminate{}包裹“完成”。请你务必只用terminate{...}包裹继续或完成。\n"
        "请详细说明你的理由（boxed和terminate都要输出，理由附在后面）。\n"
        "互动动作/组件名: " + str(interact_name)
    )
    image_content = [{"type": "text", "text": prompt_base}]
    for idx, img in enumerate(img_b64_list):
        image_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + img},
        })
    try:
        response = get_response([{"role": "user", "content": image_content}])
        output = str(response.choices[0].message)
        print(output)
        print(f"完成检测")
        return output, compare_messages
    except Exception as ex:
        print(f"Exception: {ex}")
        return ""

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Killing pid={proc.pid}, name={proc.name()}, port={port}")
                    for child in proc.children(recursive=True):
                        print(f"Killing child pid={child.pid}, name={child.name()}")
                        child.kill()
                    proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        except Exception as e:
            print(f"Error checking proc {proc.pid}: {e}")

def clean_text(text):
    if not text:
        return ''
    return ''.join(c if c.isalnum() or c in ('-_') else '_' for c in text.strip())

def wait_server_up(url, timeout=40):
    start_time = time.time()
    while True:
        try:
            res = requests.get(url, timeout=1)
            if res.status_code == 200:
                return True
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise TimeoutError("Server not up after {}s".format(timeout))

def get_current_img_b64(port):
    r = requests.get(f"http://localhost:{port}/observe_sized").json()
    return r["image_b64"]

def domtree_hash(domtree):
    domtree_str = json.dumps(domtree, sort_keys=True)
    return hashlib.md5(domtree_str.encode('utf-8')).hexdigest()

def save_image(img_b64, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(img_b64))

def deep_diff(a, b):
    return json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)

def collect_nodes_by_tag(node, tags):
    res = []
    if "tag" in node and node["tag"].lower() in tags:
        res.append({
            "id": node["id"],
            "tag": node["tag"].lower(),
            "attrs": node.get('attrs', {})
        })
    for child in node.get("children", []):
        res.extend(collect_nodes_by_tag(child, tags))
    return res

def collect_nodes_by_id(node, ids):
    res = []
    if "id" in node and node["id"] in ids:
        res.append({
            "id": node["id"],
            "attrs": node.get('attrs', {}),
            "visible_text": node.get('visible_text', "")
        })
    for child in node.get("children", []):
        res.extend(collect_nodes_by_id(child, ids))
    return res

def collect_button_ids(node, button_text_visited=None, no_text_attrs_visited=None):
    if button_text_visited is None:
        button_text_visited = set()
    if no_text_attrs_visited is None:
        no_text_attrs_visited = set()
    res = []
    children = node.get("children", [])
    for child in children:
        if not "tag" in child:
            continue
        tag = child["tag"].lower()
        if tag == "button" or tag == "a" or (tag == "input" and child["attrs"].get("type", "").lower() in ("button", "submit")):
            txt = (child.get("attrs", {}).get("text", None) or child.get("visible_text", None) or "").strip()
            can_interact = child.get("can_interact", True)
            if can_interact:
                if txt:
                    if txt not in button_text_visited:
                        button_text_visited.add(txt)
                        res.append({
                            "id": child["id"], "text": txt, "tag": tag, "attrs": child.get("attrs", {}), "can_interact": can_interact
                        })
                else:
                    attrs_set = frozenset(child.get("attrs", {}).items())
                    if attrs_set not in no_text_attrs_visited:
                        no_text_attrs_visited.add(attrs_set)
                        res.append({
                            "id": child["id"], "tag": tag, "attrs": child.get("attrs", {}), "can_interact": can_interact
                        })
        res.extend(collect_button_ids(child, button_text_visited, no_text_attrs_visited))
    return res

def get_node_text(node):
    texts = []
    if "text" in node:
        texts.append(node["text"])
    for c in node.get("children", []):
        texts.append(get_node_text(c))
    return ''.join(texts)

def collect_select_values(node):
    results = []
    def dfs(n):
        if "tag" in n and n["tag"].lower() == "select":
            values = []
            for c in n.get("children", []):
                if c.get("tag", "").lower() == "option":
                    v = c.get("attrs", {}).get("value")
                    if v is None or v == "":
                        v = get_node_text(c)
                    values.append(v)
            results.append({"id": n["id"], "values": values})
        for c in n.get("children", []):
            dfs(c)
    dfs(n=node)
    return results

def set_page_state(state, port):
    requests.post(f"http://localhost:{port}/reset").json()
    time.sleep(5)
    for step in state["actions"]:
        action = step["action"]
        dt_info = requests.get(f"http://localhost:{port}/dom_tree_with_id").json()
        domtree = dt_info["domtree"]
        id2xpath = dt_info["id2xpath"]
        if action == "click":
            resp = requests.post(
                f"http://localhost:{port}/click",
                json={"id": step["id"], "id2xpath": id2xpath}
            ).json()
        elif action == "input":
            resp = requests.post(
                f"http://localhost:{port}/enter",
                json={"id": step["id"], "text": step["input_val"], "id2xpath": id2xpath}
            ).json()
        elif action == "select":
            resp = requests.post(
                f"http://localhost:{port}/select",
                json={"id": step["id"], "value": step["value"], "id2xpath": id2xpath}
            ).json()
        else:
            pass
        if not resp:
            return False
    time.sleep(0.5)
    return True

def set_page_state_for_each_action(state, port):
    img_b64_list = []
    resp = []
    for step in state["actions"]:
        action = step["action"]
        dt_info = requests.get(f"http://localhost:{port}/dom_tree_with_id").json()
        domtree = dt_info["domtree"]
        id2xpath = dt_info["id2xpath"]
        if action == "click":
            res = requests.post(
                f"http://localhost:{port}/click",
                json={"id": step["id"], "id2xpath": id2xpath}
            ).json()
            resp.append(res)
            img_b64 = get_current_img_b64(port)
            img_b64_list.append({"action": step, "img_b64": img_b64})
        elif action == "input":
            res = requests.post(
                f"http://localhost:{port}/enter",
                json={"id": step["id"], "text": step["input_val"], "id2xpath": id2xpath}
            ).json()
            resp.append(res)
            img_b64 = get_current_img_b64(port)
            img_b64_list.append({"action": step, "img_b64": img_b64})
        elif action == "select":
            res = requests.post(
                f"http://localhost:{port}/select",
                json={"id": step["id"], "value": step["value"], "id2xpath": id2xpath}
            ).json()
            resp.append(res)
            img_b64 = get_current_img_b64(port)
            img_b64_list.append({"action": step, "img_b64": img_b64})
        else:
            pass
    return img_b64_list, resp

def extract_boxed_actions(text):
    lst = re.findall(r'\\+boxed\{([^}]*)\}', text)
    filtered = []
    for act in lst:
        if not act:
            continue
        filtered.append(act)
    return filtered

def smart_split(s, sep=','):
    res = []
    buf = ''
    level = 0
    for c in s:
        if c == '[':
            level += 1
        elif c == ']':
            level -= 1
        if c == sep and level == 0:
            res.append(buf.strip())
            buf = ''
        else:
            buf += c
    if buf:
        res.append(buf.strip())
    return res

def parse_action_seq(raw_action_seq):
    def strip_quotes(s):
        return re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', '', s)
    actions = []
    for act in smart_split(raw_action_seq):
        act = act.strip()
        if not act:
            continue
        m = re.match(r'click\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            actions.append({"action": "click", "id": id_})
            continue
        m = re.match(r'enter\[(.*?)\]\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            val = strip_quotes(m.group(2).strip())
            actions.append({"action": "input", "id": id_, "input_val": val})
            continue
        m = re.match(r'select\[(.*?)\]\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            val = strip_quotes(m.group(2).strip())
            actions.append({"action": "select", "id": id_, "value": val})
            continue
        return None
    return actions

def get_op_actions_by_model_no_history(img_b64, domtree, messages, messages_to_get_history_clicked, domtree_history):
    history_actions = []
    for idx, msg in enumerate(messages_to_get_history_clicked):
        if msg.get('role') == 'assistant':
            msg_id = msg.get('msg_id', idx)
            boxed_raws = extract_boxed_actions(msg.get('content', ''))
            actions_per_seq = [r for r in (parse_action_seq(bx) for bx in boxed_raws) if r is not None]
            for actlist in actions_per_seq:
                for act in actlist:
                    history_actions.append({'act': act, 'msg_id': msg_id})

    history_elements_desc = []
    for item in history_actions:
        act = item['act']
        domtree_for_search = domtree_history.get(item['msg_id'])
        node = find_node_by_id_for_prompt(domtree_for_search, act['id'])
        desc = str(node)
        if desc.strip():
            history_elements_desc.append(desc)
    history_info_prompt = ""
    if history_elements_desc:
        history_info_prompt = "\n".join(history_elements_desc) + "\n"

    prompt_text = f"""你是一位互动网页助手，我现在希望检测这个网页中所有可互动按钮是否都可以正常工作，例如，如果页面中有搜索框，请搜索一个合理的内容，并点击确认，期望页面会发生变化。请注意，如果多个互动组件几乎一模一样，请只从中选一个。比如页面中有多个相似的条目，每个条目都有“编辑”按钮，请只选择一次。
        现在的页面状态处于检测过程中的一环，我会发送给你现在已经点击了哪些组件。如果你发现之前点击过别的图片，请专注于这次发给你的图片与之前的有何不同，例如弹出了某个新窗口，请一定只选择新部分中的互动组件。如果你发现这次给你的图片与历史中的某次图片几乎一致，例如按钮全部一致，只有微小的文字差异；那请直接回复“本页面所有操作已完成”
        注意，如果两个互动按钮间并非连续关系，比如在同页面下的两个点击按钮，请每个boxed中只包含一个，将它们分开。如果是连续关系，比如输入多个内容并点击确认，请把它们放在同一个boxed内。将你的答案分别用latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。 【历史点击过的dom element】 {history_info_prompt}
        【注意】：请只返回答案！不要有多余的东西！
        页面信息:\n{domtree}\n,"""
    print(f"最新请求中的历史domtree:{history_info_prompt}")
    image_content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": "data:image/jpg;base64," + img_b64}}
    ]
    for ex in messages:
        if ex["role"] == "user":
            ex["content"][0]["text"] = """你是一位互动网页助手，我现在希望检测这个网页中所有可互动按钮是否都可以正常工作，例如，如果页面中有搜索框，请搜索一个合理的内容，并点击确认，期望页面会发生变化。请注意，如果多个互动组件几乎一模一样，请只从中选一个。比如页面中有多个相似的条目，每个条目都有“编辑”按钮，请只选择一次。
        现在的页面状态处于检测过程中的一环，请检查我与你的互动历史来判断现在已经点击了哪些组件。如果你发现之前点击过别的图片，请专注于这次发给你的图片与之前的有何不同，例如弹出了某个新窗口，请一定只选择新部分中的互动组件。如果你发现这次给你的图片与历史中的某次图片几乎一致，例如按钮全部一致，只有微小的文字差异；那请直接回复“本页面所有操作已完成”
        注意，如果两个互动按钮间并非连续关系，比如在同页面下的两个点击按钮，请每个boxed中只包含一个，将它们分开。如果是连续关系，比如输入多个内容并点击确认，请把它们放在同一个boxed内。将你的答案分别用latex的\\boxed{}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。"""

    print_messages_for_print = []
    for ex in messages:
        new_ex = copy.deepcopy(ex)
        if new_ex["role"] == "user":
            del new_ex["content"][1]
        print_messages_for_print.append(new_ex)
    print("正在打印messages:", print_messages_for_print)
    print("正在打印messages len:", len(str(print_messages_for_print)))

    msg_id = len(messages)
    messages.append({"role": "user", "content": image_content})
    messages_to_get_history_clicked.append({"role": "user", "content": image_content, "msg_id": msg_id})
    response= get_response(messages)
    output = str(response.choices[0].message)
    messages.append({"role": "assistant", "content": output})
    messages_to_get_history_clicked.append({"role": "assistant", "content": output, "msg_id": msg_id})
    domtree_history[msg_id] = copy.deepcopy(domtree)
    boxed_raws = extract_boxed_actions(output)
    print(output)
    actions_per_seq = [r for r in (parse_action_seq(bx) for bx in boxed_raws) if r is not None]
    print("模型动作推荐 boxed:", actions_per_seq)
    return actions_per_seq, output

def explore_html(html_file, port):
    step_records = []

    def _dump_partial_result():
        try:
            compare_records = [r for r in step_records if r.get("step_type") == "compare"]
            total_trials = len(compare_records)
            total_compare_records = [
                ",".join([a.get('action', '') + str(a.get('id', '')) for a in r.get("actions_seq", [])])
                for r in compare_records
            ]
            pass_records = [r for r in compare_records if r.get("pass")]
            pass_num = len(pass_records)
            pass_texts = [
                ",".join([a.get('action', '') + str(a.get('id', '')) for a in r.get("actions_seq", [])])
                for r in pass_records
            ]
            pass_rate = float(pass_num) / total_trials if total_trials > 0 else 0.0

            def node_signature(node):
                if node is None:
                    return "none"
                return f"{node.get('id','')}_{node.get('tag','')}_{json.dumps(node.get('attrs',{}),ensure_ascii=False,sort_keys=True)}_{node.get('visible_text','')}"

            all_action_dom_elements = []
            pass_action_dom_elements = []
            dom_sig_set_all = set()
            dom_sig_set_pass = set()

            for rec in compare_records:
                for a in rec.get("actions_seq", []):
                    dom_info = a.get("dom_info")
                    sig = node_signature(dom_info)
                    all_action_dom_elements.append(dom_info)
                    dom_sig_set_all.add(sig)
                if rec.get("pass"):
                    for a in rec.get("actions_seq", []):
                        dom_info = a.get("dom_info")
                        sig = node_signature(dom_info)
                        pass_action_dom_elements.append(dom_info)
                        dom_sig_set_pass.add(sig)

            result_json = {
                "html_file": html_file,
                "steps": step_records,
                "total_compare_steps": total_trials,
                "total_compare_records": total_compare_records,
                "pass_num": pass_num,
                "pass_texts": pass_texts,
                "pass_rate": pass_rate,
                "all_action_dom_elements": all_action_dom_elements,
                "pass_action_dom_elements": pass_action_dom_elements
            }

            out_json_path = os.path.join(f"./{IMAGE_DIR}/", f"{os.path.splitext(os.path.basename(html_file))[0]}.json")
            os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 已保存当前进度到 {out_json_path}")
        except Exception as ee:
            print(f"[WARN] 写入部分结果失败: {ee}")

    p = None
    try:
        html_name = os.path.splitext(os.path.basename(html_file))[0]
        os.makedirs(f"./{IMAGE_DIR}/{html_name}", exist_ok=True)
        os.makedirs(BUG_DIR, exist_ok=True)
        kill_process_on_port(port)
        p = subprocess.Popen(
            [sys.executable, "./webenv-init/webenv.py", html_file, '--port', str(port)]
        )
        wait_server_up(f"http://localhost:{port}/dom_tree_with_id")
        dt = requests.get(f"http://localhost:{port}/dom_tree_with_id").json()
        domtree = dt["domtree"]
        id2xpath = dt["id2xpath"]
        initial_hash = domtree_hash(domtree)
        all_layers = []
        visited_hashes = set()
        visited_hashes.add(initial_hash)
        init_img = get_current_img_b64(port)
        start_name = "start"
        layer0 = [{
            "name": start_name,
            "seq_name": "",
            "text": "start",
            "domtree": domtree,
            "domtree_hash": initial_hash,
            "id2xpath": id2xpath,
            "actions": [],
            "img_b64": init_img,
            "messages_chain": []
        }]
        all_layers.append(layer0)
        save_image(get_current_img_b64(port), f"./{IMAGE_DIR}/{html_name}/start.png")
        max_depth = 5

        messages = []
        messages_to_get_history_clicked = []
        domtree_history = {}
        domtree_history[0] = copy.deepcopy(domtree)

        for depth in range(max_depth):
            layer = all_layers[depth]
            next_layer = []
            for count_state in layer:
                count_name = count_state["name"]
                print(f"发现state {count_name} in layer ")
            for idx, state in enumerate(layer):
                set_page_state(state, port)
                dt_info = requests.get(f"http://localhost:{port}/dom_tree_with_id").json()
                domtree = dt_info["domtree"]
                id2xpath = dt_info["id2xpath"]
                button_nodes = collect_button_ids(domtree)
                input_nodes = collect_nodes_by_tag(domtree, ["input", "textarea"])
                select_nodes_info = collect_select_values(domtree)
                can_operate = bool(button_nodes or input_nodes or select_nodes_info)
                if can_operate:
                    name = state["name"]
                    print(f"本次请求为第{depth}层请求，原图为{name}")
                    actions_seqs, model_raw_output = get_op_actions_by_model_no_history(
                        state["img_b64"], domtree, messages, messages_to_get_history_clicked, domtree_history
                    )
                    detailed_action_seqs = []
                    for actlist in actions_seqs:
                        detailed_actlist = []
                        for a in actlist:
                            dom_info = find_node_by_id_for_prompt(domtree, a.get("id"))
                            a2 = dict(a)
                            a2['dom_info'] = dom_info
                            detailed_actlist.append(a2)
                        detailed_action_seqs.append(detailed_actlist)

                    step_records.append({
                        "step_type": "list",
                        "img_name": f"{state['name']}",
                        "domtree": str(domtree),
                        "actions_seqs": [dict(a) for actlist in detailed_action_seqs for a in actlist],
                        "actions_seqs_detail": detailed_action_seqs,
                        "model_output": model_raw_output,
                        "step_idx": len(step_records),
                    })

                    for seq_idx, actions in enumerate(actions_seqs):
                        act_ids = []
                        for a in actions:
                            act_ids.append(a["id"])
                        elements = collect_nodes_by_id(domtree, act_ids)
                        temp_state = {"actions": state["actions"]}
                        set_page_state(temp_state, port)
                        new_action_state = {"actions": actions}
                        new_action_img_list, click_t_or_f = set_page_state_for_each_action(new_action_state, port)
                        print("本次操作结果：", click_t_or_f)
                        for res in click_t_or_f:
                            if not res["result"]:
                                cp_bad_html(html_file)
                        combined_actions = state["actions"] + actions
                        texts = []
                        extracted_text = []
                        img_b64_list = []
                        img_name_list = []
                        texts.append(state["seq_name"])
                        img_b64_list.append(state["img_b64"])
                        print("正在准备请求：", state["name"])
                        for step_action in new_action_img_list:
                            t = ""
                            a = step_action["action"]
                            print("找到action:", a)
                            if a["action"] == "input":
                                t = f"{a['action']}_{a['id']}" + "_" + a["input_val"]
                            if a["action"] == "select":
                                t = f"{a['action']}_{a['id']}" + "_" + a["value"]
                            if a["action"] == "click":
                                node = find_node_by_id_for_prompt(domtree, a["id"])
                                if node:
                                    t = node.get("visible_text") or node.get("attrs", {}).get("text") or node.get("attrs", {}).get("value") or node.get("attrs", {}).get("placeholder") or a.get("input_val") or a.get("value") or ""
                            texts.append(clean_text(t) if t else f"{a['action']}_{a['id']}")
                            extracted_text.append(clean_text(t) if t else f"{a['action']}_{a['id']}")
                            seq_name = "_".join(texts) if texts else f"seq_{seq_idx}"
                            extracted_name = "_".join(extracted_text) if extracted_text else f"some_where_in_the_picture"
                            img_b64_list.append(step_action["img_b64"])
                            imgB_file = f"./{IMAGE_DIR}/{html_name}/{seq_name}.png"
                            print("正在准备请求：", imgB_file)
                            save_image(step_action["img_b64"], imgB_file)
                            img_name_list.append(imgB_file)
                        interact_names = []
                        for a in actions:
                            if a['action'] == 'click': interact_names.append(f"click[{a['id']}]")
                            elif a['action'] == 'input': interact_names.append(f"enter[{a['id']}][{a.get('input_val','')}]")
                            elif a['action'] == 'select': interact_names.append(f"select[{a['id']}][{a.get('value','')}]")
                        interact_name_str = ','.join(interact_names)
                        previous_messages_chain = state.get("messages_chain", [])
                        output, new_messages_chain = get_compare_response(img_b64_list, extracted_name, previous_messages_chain)
                        termination_flag = None
                        terminate_matches = re.findall(r'terminate\{(.*?)\}', output)
                        if terminate_matches:
                            termination_text = terminate_matches[-1].strip()
                        else:
                            termination_text = ""
                        terminate_flag = termination_text == "继续"
                        compare_result = extract_boxed_text(output)
                        final_result = check_yes_no(compare_result)
                        pass_flag = final_result is True
                        imgB = get_current_img_b64(port)

                        actions_detail = []
                        for a in actions:
                            dom_info = find_node_by_id_for_prompt(domtree, a.get("id"))
                            a2 = dict(a)
                            a2["dom_info"] = dom_info
                            actions_detail.append(a2)

                        step_records.append({
                            "step_type": "compare",
                            "img_name_before": f"{state['name']}",
                            "img_name_list_after": img_name_list,
                            "domtree": str(domtree),
                            "actions_seq": actions_detail,
                            "component_info": elements,
                            "model_output": model_raw_output,
                            "detect_output": output,
                            "pass": pass_flag,
                            "step_idx": len(step_records),
                        })
                        if pass_flag:
                            dt2 = requests.get(f"http://localhost:{port}/dom_tree_with_id").json()
                            next_domtree = dt2["domtree"]
                            next_hash = domtree_hash(next_domtree)
                            if terminate_flag:
                                print(f"已经写入下一轮layer")
                                state_item = {
                                    "name": imgB_file,
                                    "seq_name": seq_name,
                                    "domtree": next_domtree,
                                    "domtree_hash": next_hash,
                                    "id2xpath": id2xpath,
                                    "actions": combined_actions,
                                    "img_b64": imgB,
                                    "messages_chain": new_messages_chain
                                }
                                next_layer.append(state_item)
                                visited_hashes.add(next_hash)
                            else:
                                print(f"动作成功完成，页面没有新内容弹出: {imgB_file}，完成保存，不计入下一轮")
                        else:
                            print(f"动作没有显著变化: {imgB_file}，跳过保存。")
                else:
                    pass
            if not next_layer:
                break
            else:
                all_layers.append(next_layer)

        _dump_partial_result()

        if p is not None:
            p.terminate()
    except Exception as e:
        print(f"Exception in explore_html({html_file}): {e}")
        # cp_bad_html(html_file)
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
        kill_process_on_port(port)

NUM_PORT = 30
PORT_BASE = 8000
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def worker(html_list, port, progress_q, model_name, image_dir, bug_dir):
    # 在子进程里更新全局，避免 spawn 导致的全局不同步
    global MODEL_NAME, IMAGE_DIR, BUG_DIR
    MODEL_NAME = model_name
    IMAGE_DIR = image_dir
    BUG_DIR = bug_dir

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(BUG_DIR, exist_ok=True)

    for html in html_list:
        html_name = os.path.basename(html)
        json_log_path = os.path.join(f"./{IMAGE_DIR}/", f"{os.path.splitext(html_name)[0]}.json")
        if os.path.exists(json_log_path):
            print(f"[SKIP] 已存在日志: {json_log_path}，跳过 {html}")
            progress_q.put(1)
            continue
        explore_html(html, port)
        progress_q.put(1)


# ===== 新增：安全的模型名到子目录 =====
def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.@" else "_" for c in s)


def run_once_for_model(model_name: str):
    # 不再建 images/ 与 bad_html/ 子目录，直接用 模型名 子文件夹
    model_dir = os.path.join(ROOT_OUTPUT_DIR, safe_name(model_name))
    image_dir = model_dir
    bug_dir = model_dir
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n========== 开始运行模型：{model_name} ==========")
    print(f"输出目录：{model_dir}")

    html_files = [
        os.path.join(input_dir, fn)
        for fn in os.listdir(input_dir)
        if fn.endswith('.html')
    ]
    groups = chunkify(html_files, NUM_PORT)
    ports = [PORT_BASE + i for i in range(NUM_PORT)]
    progress_q = Queue()
    total = len(html_files)
    pbar = tqdm(total=total, desc=f'全部HTML ({model_name})', ncols=80)
    procs = []
    for group, port in zip(groups, ports):
        p = Process(target=worker, args=(group, port, progress_q, model_name, image_dir, bug_dir))
        p.start()
        procs.append(p)
        time.sleep(3)
    finished = 0
    while finished < total:
        n = progress_q.get()
        finished += n
        pbar.update(n)
    for p in procs:
        p.join()
    pbar.close()
    print(f"========== 模型完成：{model_name} ==========\n")

if __name__ == '__main__':
    # 依次跑每个模型
    for m in MODELS:
        run_once_for_model(m)
