import os
import argparse
import json
import requests
import base64
import time
import subprocess
import psutil
import hashlib
import sys
import re
import copy
import ast
import shutil
from tqdm import tqdm
from multiprocessing import Process, Queue
import openai
from openai import OpenAI


# ------------------------------
# 1. 加载配置：支持命令行与配置文件
# ------------------------------
def load_config():
    parser = argparse.ArgumentParser(description="WEBVIA Exploration Pipeline (HTML / URL)")
    parser.add_argument("--config", type=str, default=None, help="路径配置文件(config.json)")
    # 通用
    parser.add_argument("--image_dir", type=str, help="输出图像与日志目录")
    parser.add_argument("--bug_dir", type=str, help="异常HTML保存目录（仅HTML模式使用）")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key")
    parser.add_argument("--api_base", type=str, help="OpenAI Base URL")
    parser.add_argument("--model_name", type=str, help="使用的模型名称")
    parser.add_argument("--num_port", type=int, help="并行端口数")
    parser.add_argument("--port_base", type=int, help="基础端口号")
    parser.add_argument("--input_type", type=str, choices=["html", "url"], help="输入类型：html 或 url")

    # HTML 模式
    parser.add_argument("--input_dir", type=str, help="输入HTML文件目录")

    # URL 模式
    parser.add_argument("--url_file", type=str, help="包含URL列表的文本文件（每行一个）")

    args = parser.parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    # 命令行优先级高于配置文件
    def get(key, default=None):
        val = getattr(args, key, None)
        if val is not None:
            return val
        return config.get(key, default)

    # 统一路径格式
    def abs_path(p):
        if not p:
            return None
        return os.path.abspath(os.path.expanduser(p))

    final_config = {
        "input_type": get("input_type", "html"),
        "input_dir": abs_path(get("input_html_dir", "./input_html")),
        "url_file": abs_path(get("url_file", "./urls.txt")),
        "image_dir": abs_path(get("image_dir", "./output_images")),
        "bug_dir": abs_path(get("bug_dir", "./trash_dir")),
        "api_key": get("api_key", ""),
        "api_base": get("api_base", ""),
        "model_name": get("model_name", "o4-mini-2025-04-16"),
        "webenv_py": get("webenv_path", "./webenv-init/webenv.py"),
        "num_port": int(get("num_port", 30)),
        "port_base": int(get("port_base", 8000)),
    }
    # print(final_config)
    return final_config


class ConfigError(Exception):
    pass


def url_to_name(url: str) -> str:
    """把任意 URL 转成短而安全的文件名片段"""
    return re.sub(r'\W+', '_', url)[:40]

def _is_writable_dir(path: str) -> tuple[bool, str | None]:
    """
    返回 (是否可写, 失败原因)
    """
    try:
        os.makedirs(path, exist_ok=True)
        testfile = os.path.join(path, ".write_test___")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testfile)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def validate_config(cfg: dict) -> dict:
    """
    校验必要项；若有问题，统一收集并一次性抛错（逐条指出）。
    返回经过规范化（绝对路径等）的 cfg。
    """
    errors = []
    # print(cfg)
    # 统一绝对路径
    def abs_path(p):
        if p is None:
            return None
        return os.path.abspath(os.path.expanduser(p))

    cfg["image_dir"] = abs_path(cfg.get("image_dir"))
    cfg["bug_dir"]   = abs_path(cfg.get("bug_dir"))  # 仅 HTML 用，但先校验可写
    cfg["input_dir"] = abs_path(cfg.get("input_dir"))
    cfg["url_file"]  = abs_path(cfg.get("url_file"))
    webenv_py = abs_path(cfg.get("webenv_py"))

    # 可选：webenv 脚本存在性（你的代码依赖这个文件）
    if not os.path.isfile(webenv_py):
        errors.append(f"[webenv_init] 缺少环境脚本：{webenv_py}（请确认路径或调整启动命令）")
    else:
        cfg["webenv_py"] = webenv_py  # 供后续使用

    # 2) 模型与 OpenAI 相关
    model_name = cfg.get("model_name", "")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append(f"[model_name] 模型名称不能为空（实得：{model_name!r}）")

    api_key = cfg.get("api_key", "")
    if not isinstance(api_key, str) or not api_key.strip():
        errors.append("[api_key] 不能为空（建议通过 --api_key 或 config.json 提供）")

    api_base = cfg.get("api_base", "")
    if not isinstance(api_base, str) or not api_base.strip():
        errors.append("[api_base] 不能为空（例如：https://api.openai.com/v1）")

    # 3) 端口设置
    try:
        num_port = int(cfg.get("num_port", 0))
        if num_port < 1:
            errors.append(f"[num_port] 必须为 >= 1 的整数（实得：{cfg.get('num_port')!r}）")
    except Exception:
        errors.append(f"[num_port] 非法整数：{cfg.get('num_port')!r}")

    try:
        port_base = int(cfg.get("port_base", -1))
        if not (1 <= port_base <= 65535):
            errors.append(f"[port_base] 必须位于 1..65535（实得：{cfg.get('port_base')!r}）")
        else:
            # 端口区间不应越界
            last_port = port_base + int(cfg.get("num_port", 0)) - 1
            if last_port > 65535:
                errors.append(
                    f"[port_base + num_port] 端口区间越界：{port_base}..{last_port}（最大65535）"
                )
    except Exception:
        errors.append(f"[port_base] 非法整数：{cfg.get('port_base')!r}")

    # 4) 按 input_type 做差异化校验
    input_type = (cfg.get("input_type") or "html").lower()
    if input_type not in ("html", "url"):
        errors.append(f"[input_type] 仅支持 'html' 或 'url'（实得：{cfg.get('input_type')!r}）")
    else:
        if input_type == "html":
            if not cfg["input_dir"] or not os.path.isdir(cfg["input_dir"]):
                errors.append(f"[input_dir] 目录不存在或不可访问：{cfg.get('input_dir')!r}")
            else:
                html_files = [fn for fn in os.listdir(cfg["input_dir"]) if fn.endswith(".html")]
                if len(html_files) == 0:
                    errors.append(f"[input_dir] 目录存在但没有任何 .html 文件：{cfg['input_dir']}")
            # bug_dir 在 HTML 模式下需要可写
            if not cfg["bug_dir"]:
                errors.append("[bug_dir] 未提供异常HTML保存目录路径")
            else:
                ok, reason = _is_writable_dir(cfg["bug_dir"])
                if not ok:
                    errors.append(f"[bug_dir] 无法创建/写入：{cfg['bug_dir']}（{reason}）")
        else:  # url
            if not cfg["url_file"] or not os.path.isfile(cfg["url_file"]):
                errors.append(f"[url_file] URL 列表文件不存在或不可访问：{cfg.get('url_file')!r}")
            else:
                with open(cfg["url_file"], "r", encoding="utf-8") as f:
                    urls = [ln.strip() for ln in f if ln.strip()]
                if len(urls) == 0:
                    errors.append(f"[url_file] 文件存在但未包含任何 URL：{cfg['url_file']}")

    # image_dir 通用校验
    if not cfg["image_dir"]:
        errors.append("[image_dir] 未提供输出目录路径")
    else:
        ok, reason = _is_writable_dir(cfg["image_dir"])
        if not ok:
            errors.append(f"[image_dir] 无法创建/写入：{cfg['image_dir']}（{reason}）")

    # 汇总报错
    if errors:
        bullet = "\n  - "
        msg = "配置校验失败，需修正以下问题：" + bullet + bullet.join(errors)
        raise ConfigError(msg)

    return cfg


# ------------------------------
# 2. 初始化配置与全局变量
# ------------------------------
config = load_config()
config = validate_config(config)  # <<< 新增：严格校验（若失败会抛出并退出）

INIT_PATH = config["webenv_py"]           # 修正：使用校验阶段写入的键名
IMAGE_DIR = config["image_dir"]
INPUT_DIR = config.get("input_dir")
BUG_DIR   = config.get("bug_dir")
INPUT_TYPE = (config.get("input_type") or "html").lower()
URL_FILE  = config.get("url_file")

openai.api_base = config["api_base"]
openai.api_key  = config["api_key"]
MODEL_NAME = config["model_name"]

client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)

NUM_PORT  = config["num_port"]
PORT_BASE = config["port_base"]

# ---- 安全命名工具 ----
MAX_FILENAME_BYTES = 255          # 常见文件系统单个文件名上限
MAX_FULLPATH_BYTES = 4096         # 进程/OS 对全路径一般极限（保守）



# ------------------------------
# 3. 工具函数
# ------------------------------
def get_response(message):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=message,
        max_completion_tokens=10000,
        timeout=600
    )
    return response


def cp_bad_html(src_file):
    # 仅 HTML 模式使用
    try:
        basename = os.path.basename(src_file)
        copypath = os.path.join(BUG_DIR, basename)
        shutil.copy(src_file, copypath)
    except Exception:
        pass


def _utf8_len(s: str) -> int:
    return len(s.encode('utf-8'))

def build_seq_names_or_fallback(state_seq_name: str,
                                step_labels: list[str],
                                out_dir: str,
                                layer_idx: int) -> tuple[list[str], bool]:
    """
    给定 state 的已有前缀（state_seq_name）、本轮步骤标签（step_labels）与输出目录（out_dir），
    先尝试“逐步拼接”的原始命名；若任何一个文件名或全路径会超长，则整体回退为：
        {state_seq_name}_action-seq-layer{layer_idx}-{k}
    返回：(names, used_fallback)
        - names: 每一步的最终 seq_name 列表（不含扩展名）
        - used_fallback: 是否触发了统一回退
    """
    # 1) 先尝试原始增量命名
    orig_names = []
    for i in range(len(step_labels)):
        parts = []
        if state_seq_name:
            parts.append(state_seq_name)
        parts += step_labels[:i+1]
        seq_name = "_".join(p for p in parts if p)
        filename = f"{seq_name}.png"
        fullpath = os.path.join(out_dir, filename)

        if _utf8_len(os.path.basename(filename)) > MAX_FILENAME_BYTES or _utf8_len(fullpath) > MAX_FULLPATH_BYTES:
            # 任何一步超限则整段回退
            break
        orig_names.append(seq_name)

    if len(orig_names) == len(step_labels):
        return orig_names, False

    # 2) 统一回退命名方案
    fb_names = []
    for k in range(1, len(step_labels) + 1):
        fb = f"action-seq-layer{layer_idx}-{k}"
        seq_name = "_".join([p for p in [state_seq_name, fb] if p])
        fb_names.append(seq_name)

    return fb_names, True


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
            resp = True
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




# ------------------------------
# 4. 请求模型
# ------------------------------

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
            "image_url": {"url": "data:image/jpg;base64," + img},
        })
    try:
        response = get_response([{"role": "user", "content": image_content}])
        output = str(response.choices[0].message)
        return output, compare_messages
    except Exception as ex:
        print(f"Exception: {ex}")
        return "", compare_messages


def get_op_actions_by_model_no_history(img_b64, domtree, messages, messages_to_get_history_clicked, domtree_history):
    """
    用模型生成可操作动作序列，并在prompt里包含历史已选动作的dom元素信息,
    domtree_history: 每条历史消息唯一id映射的domtree
    """
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
    image_content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": "data:image/jpg;base64," + img_b64}}
    ]
    for ex in messages:
        if ex["role"] == "user":
            ex["content"][0]["text"] = f"""你是一位互动网页助手，我现在希望检测这个网页中所有可互动按钮是否都可以正常工作，例如，如果页面中有搜索框，请搜索一个合理的内容，并点击确认，期望页面会发生变化。请注意，如果多个互动组件几乎一模一样，请只从中选一个。比如页面中有多个相似的条目，每个条目都有“编辑”按钮，请只选择一次。
        现在的页面状态处于检测过程中的一环，请检查我与你的互动历史来判断现在已经点击了哪些组件。如果你发现之前点击过别的图片，请专注于这次发给你的图片与之前的有何不同，例如弹出了某个新窗口，请一定只选择新部分中的互动组件。如果你发现这次给你的图片与历史中的某次图片几乎一致，例如按钮全部一致，只有微小的文字差异；那请直接回复“本页面所有操作已完成”
        注意，如果两个互动按钮间并非连续关系，比如在同页面下的两个点击按钮，请每个boxed中只包含一个，将它们分开。如果是连续关系，比如输入多个内容并点击确认，请把它们放在同一个boxed内。将你的答案分别用latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。"""
    print_messages_for_print = []
    for ex in messages:
        new_ex = copy.deepcopy(ex)
        if new_ex["role"] == "user":
            del new_ex["content"][1]
        print_messages_for_print.append(new_ex)

    msg_id = len(messages)
    messages.append({"role": "user", "content": image_content})
    messages_to_get_history_clicked.append({"role": "user", "content": image_content, "msg_id": msg_id})
    response= get_response(messages)
    output = str(response.choices[0].message)
    messages.append({"role": "assistant", "content": output})
    messages_to_get_history_clicked.append({"role": "assistant", "content": output, "msg_id": msg_id})
    domtree_history[msg_id] = copy.deepcopy(domtree)
    boxed_raws = extract_boxed_actions(output)
    actions_per_seq = [r for r in (parse_action_seq(bx) for bx in boxed_raws) if r is not None]
    return actions_per_seq, output


# ---------- 共用的结果写盘 ----------
def _dump_partial_result_common(step_records, out_json_path, meta_fields: dict):
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
            **meta_fields,
            "steps": step_records,
            "total_compare_steps": total_trials,
            "total_compare_records": total_compare_records,
            "pass_num": pass_num,
            "pass_texts": pass_texts,
            "pass_rate": pass_rate,
            "all_action_dom_elements": all_action_dom_elements,
            "pass_action_dom_elements": pass_action_dom_elements
        }

        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已保存当前进度到 {out_json_path}")
    except Exception as ee:
        print(f"[WARN] 写入部分结果失败: {ee}")





# ------------------------------
# 5. 探索主函数
# ------------------------------

# ---------- HTML 模式 ----------
def explore_html(html_file, port):
    step_records = []

    p = None
    try:
        html_name = os.path.splitext(os.path.basename(html_file))[0]
        os.makedirs(f"{IMAGE_DIR}/{html_name}", exist_ok=True)
        os.makedirs(BUG_DIR, exist_ok=True)
        kill_process_on_port(port)
        p = subprocess.Popen(
            [sys.executable, INIT_PATH, html_file, '--port', str(port)]
        )
        wait_server_up(f"http://localhost:{port}/dom_tree_with_id")
        dt = requests.get(f"http://localhost:{port}/dom_tree_with_id").json()
        domtree = dt["domtree"]
        id2xpath = dt["id2xpath"]
        initial_hash = domtree_hash(domtree)
        all_layers = []
        visited_hashes = {initial_hash}
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
        save_image(get_current_img_b64(port), f"{IMAGE_DIR}/{html_name}/start.png")
        max_depth = 5

        messages = []
        messages_to_get_history_clicked = []
        domtree_history = {0: copy.deepcopy(domtree)}

        for depth in range(max_depth):
            layer = all_layers[depth]
            next_layer = []
            for count_state in layer:
                print(f"发现state {count_state['name']} in layer ")
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
                    print(f"本次请求为第{depth}层请求，原图为{state['name']}")
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
                        act_ids = [a["id"] for a in actions]
                        elements = collect_nodes_by_id(domtree, act_ids)
                        temp_state = {"actions": state["actions"]}
                        set_page_state(temp_state, port)
                        new_action_state = {"actions": actions}
                        new_action_img_list, click_t_or_f = set_page_state_for_each_action(new_action_state, port)
                        for res in click_t_or_f:
                            if isinstance(res, dict) and (not res.get("result", True)):
                                cp_bad_html(html_file)

                        combined_actions = state["actions"] + actions
                        texts, extracted_text = [], []
                        img_b64_list, img_name_list = [], []
                        texts.append(state["seq_name"])
                        img_b64_list.append(state["img_b64"])
                        print("正在准备检验请求：", state["name"])
                        step_labels = []        # 用于文件命名的清洗后标签（texts 的增量）
                        extracted_labels = []   # 用于 extracted_name 的清洗后标签
                        step_imgs_b64 = []      # 每步的 base64

                        for step_action in new_action_img_list:
                            t = ""
                            a = step_action["action"]
                            if a["action"] == "input":
                                t = f"{a['action']}_{a['id']}" + "_" + a["input_val"]
                            if a["action"] == "select":
                                t = f"{a['action']}_{a['id']}" + "_" + a["value"]
                            if a["action"] == "click":
                                node = find_node_by_id_for_prompt(domtree, a["id"])
                                if node:
                                    t = node.get("visible_text") or node.get("attrs", {}).get("text") or \
                                        node.get("attrs", {}).get("value") or node.get("attrs", {}).get("placeholder") or \
                                        a.get("input_val") or a.get("value") or ""
                            label = clean_text(t) if t else f"{a['action']}_{a['id']}"
                            step_labels.append(label)
                            extracted_labels.append(label)
                            step_imgs_b64.append(step_action["img_b64"])

                        # --- 统一检查是否会出现“文件名或路径过长”，必要时整体回退 ---
                        out_dir_for_this = f"{IMAGE_DIR}/{html_name}"
                        # state["seq_name"] 作为前缀延续
                        final_names, used_fallback = build_seq_names_or_fallback(
                            state_seq_name=state.get("seq_name", ""),
                            step_labels=step_labels,
                            out_dir=out_dir_for_this,
                            layer_idx=depth  # 用当前层数 depth
                        )

                        # extracted_name：若触发回退，用最后一个回退名；否则仍按原策略用拼接后的 extracted_labels
                        if used_fallback:
                            extracted_name = final_names[-1]
                        else:
                            extracted_name = "_".join(extracted_labels) if extracted_labels else "some_where_in_the_picture"

                        # --- 落盘（使用最终确定的 final_names） ---
                        img_name_list = []
                        img_b64_list = [state["img_b64"]]  # compare 需要把起始图也放进去
                        texts = [state.get("seq_name", "")] if state.get("seq_name", "") else []
                        last_image_path = ""
                        # 将每一步图片按 final_names 写出
                        for seq_name, img_b64_one in zip(final_names, step_imgs_b64):
                            img_path = os.path.join(out_dir_for_this, f"{seq_name}.png")
                            save_image(img_b64_one, img_path)
                            img_name_list.append(img_path)
                            img_b64_list.append(img_b64_one)
                            last_image_path = img_path

                        # 注意：texts 只用于延续“逻辑链前缀”，当未回退时维持原行为；若回退则 texts 不再继续增长长串，直接以最终名为后续 state['seq_name']
                        if used_fallback:
                            next_seq_chain = final_names[-1]  # 已包含 state['seq_name'] 前缀
                        else:
                            next_seq_chain = "_".join([p for p in [state.get("seq_name","")] + step_labels if p])



                        interact_names = []
                        for a in actions:
                            if a['action'] == 'click': interact_names.append(f"click[{a['id']}]")
                            elif a['action'] == 'input': interact_names.append(f"enter[{a['id']}][{a.get('input_val','')}]")
                            elif a['action'] == 'select': interact_names.append(f"select[{a['id']}][{a.get('value','')}]")
                        previous_messages_chain = state.get("messages_chain", [])
                        output, new_messages_chain = get_compare_response(img_b64_list, extracted_name, previous_messages_chain)

                        terminate_matches = re.findall(r'terminate\{(.*?)\}', output)
                        termination_text = terminate_matches[-1].strip() if terminate_matches else ""
                        terminate_flag = (termination_text == "继续")
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
                                state_item = {
                                    "name": last_image_path,
                                    "seq_name": next_seq_chain,
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
                                print(f"动作成功完成，页面没有新内容弹出: {last_image_path}，完成保存，不计入下一轮")
                        else:
                            print(f"动作没有显著变化: {last_image_path}，跳过保存。")

                # else: nothing to do
            if not next_layer:
                break
            else:
                all_layers.append(next_layer)

        out_json_path = os.path.join(f"{IMAGE_DIR}/", f"{os.path.splitext(os.path.basename(html_file))[0]}.json")
        _dump_partial_result_common(step_records, out_json_path, {"html_file": html_file})

        if p is not None:
            p.terminate()
    except Exception as e:
        print(f"Exception in explore_html({html_file}): {e}")
        cp_bad_html(html_file)
        out_json_path = os.path.join(f"{IMAGE_DIR}/", f"{os.path.splitext(os.path.basename(html_file))[0]}.json")
        _dump_partial_result_common(step_records, out_json_path, {"html_file": html_file})
    finally:
        # 兜底
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
        kill_process_on_port(port)


# ---------- URL 模式 ----------
def explore_url(url, port):
    step_records = []

    p = None
    try:
        url_name = re.sub(r'\W+', '_', url)[:40]
        image_dir = os.path.join(IMAGE_DIR, url_name)
        os.makedirs(image_dir, exist_ok=True)
        kill_process_on_port(port)

        # 与你此前的 URL 版本对齐：启动 webenv，并在启动后 load_url
        p = subprocess.Popen([sys.executable, INIT_PATH, url, '--port', str(port)])
        wait_server_up(f"http://localhost:{port}/dom_tree_with_id")
        # 再次 load_url，兼容某些实现约定（安全冗余）
        try:
            requests.post(f"http://localhost:{port}/load_url", json={'url': url})
        except Exception:
            pass
        time.sleep(2)

        dt = requests.get(f"http://localhost:{port}/dom_tree_with_id").json()
        domtree = dt["domtree"]
        id2xpath = dt["id2xpath"]
        initial_hash = domtree_hash(domtree)
        all_layers = []
        visited_hashes = {initial_hash}
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
        save_image(get_current_img_b64(port), f"{IMAGE_DIR}/{url_name}/start.png")
        max_depth = 5

        messages = []
        messages_to_get_history_clicked = []
        domtree_history = {0: copy.deepcopy(domtree)}

        for depth in range(max_depth):
            layer = all_layers[depth]
            next_layer = []
            for count_state in layer:
                print(f"发现state {count_state['name']} in layer ")
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
                    print(f"本次请求为第{depth}层请求，原图为{state['name']}")
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
                        act_ids = [a["id"] for a in actions]
                        elements = collect_nodes_by_id(domtree, act_ids)
                        temp_state = {"actions": state["actions"]}
                        set_page_state(temp_state, port)
                        new_action_state = {"actions": actions}
                        new_action_img_list, click_t_or_f = set_page_state_for_each_action(new_action_state, port)
                        combined_actions = state["actions"] + actions

                        texts, extracted_text = [], []
                        img_b64_list, img_name_list = [], []
                        texts.append(state["seq_name"])
                        img_b64_list.append(state["img_b64"])
                        print("正在准备互动请求：", state["name"])
                        # --- 先收集每一步的标签与图片，但不立刻落盘 ---
                        step_labels = []        # 用于文件命名的清洗后标签（texts 的增量）
                        extracted_labels = []   # 用于 extracted_name 的清洗后标签
                        step_imgs_b64 = []      # 每步的 base64

                        for step_action in new_action_img_list:
                            t = ""
                            a = step_action["action"]
                            if a["action"] == "input":
                                t = f"{a['action']}_{a['id']}" + "_" + a["input_val"]
                            if a["action"] == "select":
                                t = f"{a['action']}_{a['id']}" + "_" + a["value"]
                            if a["action"] == "click":
                                node = find_node_by_id_for_prompt(domtree, a["id"])
                                if node:
                                    t = node.get("visible_text") or node.get("attrs", {}).get("text") or \
                                        node.get("attrs", {}).get("value") or node.get("attrs", {}).get("placeholder") or \
                                        a.get("input_val") or a.get("value") or ""
                            label = clean_text(t) if t else f"{a['action']}_{a['id']}"
                            step_labels.append(label)
                            extracted_labels.append(label)
                            step_imgs_b64.append(step_action["img_b64"])

                        # --- 统一检查是否会出现“文件名或路径过长”，必要时整体回退 ---
                        out_dir_for_this = f"{IMAGE_DIR}/{url_name}"
                        # state["seq_name"] 作为前缀延续
                        final_names, used_fallback = build_seq_names_or_fallback(
                            state_seq_name=state.get("seq_name", ""),
                            step_labels=step_labels,
                            out_dir=out_dir_for_this,
                            layer_idx=depth  # 用当前层数 depth
                        )

                        # extracted_name：若触发回退，用最后一个回退名；否则仍按原策略用拼接后的 extracted_labels
                        if used_fallback:
                            extracted_name = final_names[-1]
                        else:
                            extracted_name = "_".join(extracted_labels) if extracted_labels else "some_where_in_the_picture"

                        # --- 落盘（使用最终确定的 final_names） ---
                        img_name_list = []
                        img_b64_list = [state["img_b64"]]  # compare 需要把起始图也放进去
                        texts = [state.get("seq_name", "")] if state.get("seq_name", "") else []
                        last_image_path = ""
                        # 将每一步图片按 final_names 写出
                        for seq_name, img_b64_one in zip(final_names, step_imgs_b64):
                            img_path = os.path.join(out_dir_for_this, f"{seq_name}.png")
                            save_image(img_b64_one, img_path)
                            img_name_list.append(img_path)
                            img_b64_list.append(img_b64_one)
                            last_image_path = img_path

                        # 注意：texts 只用于延续“逻辑链前缀”，当未回退时维持原行为；若回退则 texts 不再继续增长长串，直接以最终名为后续 state['seq_name']
                        if used_fallback:
                            next_seq_chain = final_names[-1]  # 已包含 state['seq_name'] 前缀
                        else:
                            next_seq_chain = "_".join([p for p in [state.get("seq_name","")] + step_labels if p])


                        interact_names = []
                        for a in actions:
                            if a['action'] == 'click': interact_names.append(f"click[{a['id']}]")
                            elif a['action'] == 'input': interact_names.append(f"enter[{a['id']}][{a.get('input_val','')}]")
                            elif a['action'] == 'select': interact_names.append(f"select[{a['id']}][{a.get('value','')}]")
                        previous_messages_chain = state.get("messages_chain", [])
                        output, new_messages_chain = get_compare_response(img_b64_list, extracted_name, previous_messages_chain)

                        terminate_matches = re.findall(r'terminate\{(.*?)\}', output)
                        termination_text = terminate_matches[-1].strip() if terminate_matches else ""
                        terminate_flag = (termination_text == "继续")
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
                                state_item = {
                                    "name": last_image_path,
                                    "seq_name": next_seq_chain,
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
                                print(f"动作成功完成，页面没有新内容弹出: {last_image_path}，完成保存，不计入下一轮")
                        else:
                            print(f"动作没有显著变化: {last_image_path}，跳过保存。")

            if not next_layer:
                break
            else:
                all_layers.append(next_layer)

        out_json_path = os.path.join(f"{IMAGE_DIR}/", f"{url_name}.json")
        _dump_partial_result_common(step_records, out_json_path, {"url": url})

        if p is not None:
            p.terminate()
    except Exception as e:
        print(f"Exception in explore_url({url}): {e}")
        safe = url_to_name(url)
        out_json_path = os.path.join(f"{IMAGE_DIR}/", f"{safe}.json")
        _dump_partial_result_common(step_records, out_json_path, {"url": url})
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
        kill_process_on_port(port)




# ------------------------------
# 6. 并行调度
# ------------------------------

# ---------- 调度 ----------
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def worker_html(html_list, port, progress_q):
    for html in html_list:
        html_name = os.path.basename(html)
        json_log_path = os.path.join(f"{IMAGE_DIR}/", f"{os.path.splitext(html_name)[0]}.json")
        if os.path.exists(json_log_path):
            print(f"[SKIP] 已存在日志: {json_log_path}，跳过 {html}")
            progress_q.put(1)
            continue
        explore_html(html, port)
        progress_q.put(1)


def worker_url(url_list, port, progress_q):
    for url in url_list:
        url_name = re.sub(r'\W+', '_', url)[:40]
        json_log_path = os.path.join(f"{IMAGE_DIR}/", f"{url_name}.json")
        if os.path.exists(json_log_path):
            print(f"[SKIP] 已存在日志: {json_log_path}，跳过 {url}")
            progress_q.put(1)
            continue
        explore_url(url, port)
        progress_q.put(1)


if __name__ == "__main__":
    try:
        if INPUT_TYPE == "html":
            html_files = [
                os.path.join(INPUT_DIR, fn)
                for fn in os.listdir(INPUT_DIR)
                if fn.endswith(".html")
            ]
            if not html_files:
                raise ConfigError(f"[INPUT_DIR] 无 .html 文件：{INPUT_DIR}")

            effective_workers = max(1, min(NUM_PORT, len(html_files)))
            groups = [html_files[i::effective_workers] for i in range(effective_workers)]
            ports  = [PORT_BASE + i for i in range(effective_workers)]

            progress_q = Queue()
            total = len(html_files)
            pbar = tqdm(total=total, desc="全部HTML", ncols=80)
            procs = []
            for group, port in zip(groups, ports):
                if not group:   # 跳过空组
                    continue
                p = Process(target=worker_html, args=(group, port, progress_q))
                p.start()
                procs.append(p)

            finished = 0
            while finished < total:
                finished += progress_q.get()
                pbar.update(1)

            for p in procs:
                p.join()
            pbar.close()


        else:  # url
            with open(URL_FILE, "r", encoding="utf-8") as f:
                url_list = [ln.strip() for ln in f if ln.strip()]
            effective_workers = max(1, min(NUM_PORT, len(url_list)))
            groups = [url_list[i::effective_workers] for i in range(effective_workers)]
            ports  = [PORT_BASE + i for i in range(effective_workers)]

            progress_q = Queue()
            total = len(url_list)
            pbar = tqdm(total=total, desc="全部URL", ncols=80)
            procs = []
            for group, port in zip(groups, ports):
                p = Process(target=worker_url, args=(group, port, progress_q))
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

    except ConfigError as ce:
        print(str(ce), file=sys.stderr)
        sys.exit(2)
