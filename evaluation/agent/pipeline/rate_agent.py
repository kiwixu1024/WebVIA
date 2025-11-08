import os
import json
from collections import Counter

def element_signature(elem, mode='normal'):
    if not isinstance(elem, dict):
        return None
    if mode == 'or':
        return (
            elem.get('tag', ''),
            tuple(sorted(elem.get('attrs', {}).items())),
        )
    else:
        return (
            elem.get('tag', ''),
            tuple(sorted(elem.get('attrs', {}).items())),
            elem.get('visible_text', '')
        )

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def get_valid_elems(json_obj, key):
    """
    返回元素和其signature模式（normal|or）
    """
    result = []
    elements = json_obj.get(key, [])
    for e in elements:
        if not isinstance(e, dict):
            continue
        mode = 'or' if e.get('OR', False) else 'normal'
        sig = element_signature(e, mode)
        # 保留模式信息
        result.append((sig, mode))
    return result

def target_signature_list(target_json, key):
    """
    返回target中的所有元素的两种签名（normal, or）
    """
    elements = target_json.get(key, [])
    result_normal = []
    result_or = []
    for e in elements:
        if not isinstance(e, dict):
            continue
        # normal模式
        result_normal.append(element_signature(e, mode='normal'))
        # or模式
        result_or.append(element_signature(e, mode='or'))
    return result_normal, result_or

def signature_in_target(sig, mode, target_normals, target_ors):
    # 判断sig是否在target中的对应模式签名列表里
    if mode == 'or':
        return sig in target_ors
    else:
        return sig in target_normals

def score_json(standard, target):
    # 获取标准元素和他们的签名模式
    std_all = get_valid_elems(standard, 'all_action_dom_elements')
    std_pass = get_valid_elems(standard, 'pass_action_dom_elements')

    # 获取target元素签名列表（normal和or两套）
    tgt_all_normal, tgt_all_or = target_signature_list(target, 'all_action_dom_elements')
    tgt_pass_normal, tgt_pass_or = target_signature_list(target, 'pass_action_dom_elements')

    # 1. 全面性
    found_count = sum(
        1 for (sig, mode) in std_all 
        if signature_in_target(sig, mode, tgt_all_normal, tgt_all_or)
    )
    total = len(std_all)
    full_score1 = 100 if total == 0 else 100 * found_count / total

    # 2. 准确性
    std_pass_in_tgt = [
        (sig, mode) for (sig, mode) in std_pass 
        if signature_in_target(sig, mode, tgt_all_normal, tgt_all_or)
    ]
    found_pass_count = sum(
        1 for (sig, mode) in std_pass_in_tgt
        if signature_in_target(sig, mode, tgt_pass_normal, tgt_pass_or)
    )
    total_pass = len(std_pass_in_tgt)
    full_score2 = 100 if total_pass == 0 else 100 * found_pass_count / total_pass

    # 3. 重复度（用normal模式统计重复，or模式为忽略visible_text的重复）
    counter = Counter(tgt_all_normal)
    repeats = sum(count - 1 for count in counter.values() if count > 1)
    deduction = repeats * 20
    full_score3 = max(100 - deduction, 0)

    # F1/Recall/Precision 基于全面性（tp/标准/target元素个数）
    tp = found_count
    std_size = total
    tgt_size = len(tgt_all_normal)
    recall = tp / std_size if std_size else 0
    precision = tp / tgt_size if tgt_size else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    f1_score = round(f1_score,4)

    final_score = ((full_score1 * 1.2) + (full_score2 * 1.05)  + (full_score3 * 0.75)) / 3

    return {
        '全面性': round(full_score1,2),
        '准确性': round(full_score2,2),
        '重复度': round(full_score3,2),
        '最终得分': round(final_score,2),
        '重复数量': repeats,
    }

def batch_score(standard_folder, target_folders):
    std_files = [f for f in os.listdir(standard_folder) if f.endswith('.json')]
    all_results = {}
    for target_folder in target_folders:
        print(f'\n>> 文件夹: {target_folder}')
        tgt_files = [f for f in os.listdir(target_folder) if f.endswith('.json')]
        results = []
        missing_files = []
        for fname in std_files:
            std_path = os.path.join(standard_folder, fname)
            tgt_path = os.path.join(target_folder, fname)
            try:
                standard = load_json(std_path)
            except Exception as e:
                print(f'读取标准失败 {fname}: {e}')
                continue
            if fname in tgt_files:
                try:
                    target = load_json(tgt_path)
                    score = score_json(standard, target)
                except Exception as e:
                    print(f'读取评分文件失败 {fname}: {e}')
                    score = {k: 0 for k in ['全面性','准确性','重复度','最终得分','重复数量']}
                score['文件'] = fname
                score['缺失'] = False
                results.append(score)
            else:
                score = {k: 0 for k in ['全面性','准确性','重复度','最终得分','重复数量']}
                score['文件'] = fname
                score['缺失'] = True
                results.append(score)
                missing_files.append(fname)
        for score in results:
            mark = ' [缺失]' if score['缺失'] else ''
            print(f'文件：{score["文件"]}{mark}')
            res = dict(score)
            del res['文件']
            del res['缺失']
            print(res)
            print('-'*30)
        keys = ['全面性', '准确性', '重复度', '最终得分', '重复数量']
        average = {}
        valid_scores = [item for item in results if not item['缺失']]
        for k in keys:
            if valid_scores:
                average[k] = round(sum(item[k] for item in valid_scores)/len(valid_scores),2)
            else:
                average[k] = 0
        summary = {
            "average": average,
            "缺失文件": missing_files
        }
        all_results[target_folder] = summary
        out_name = f'评分结果汇总_{os.path.basename(target_folder)}.json'
        with open(out_name, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'\n已保存当前评分平均结果到：{out_name}')
    
    out_name = 'agent_all_result.json'
    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'\n全部文件夹得分已汇总到：{out_name}')

if __name__ == '__main__':
    standard_folder = 'ground_truths'
    folder_list = [
        "ALL_model_result/o4-mini-2025-04-16",
        "ALL_model_result/claude-sonnet-4-20250514-thinking",
        "ALL_model_result/gpt-5-2025-08-07",
        "ALL_model_result/gemini-2.5-pro",
        "ALL_model_result/claude-3-7-sonnet-20250219-thinking",
        "ALL_model_result/gpt-4o-2024-11-20"
    ]
    batch_score(standard_folder, folder_list)