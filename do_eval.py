#내장 함수 관련
import json
import argparse
from typing import Dict

# 텍스트 데이터 관련
import re
import collections

def f1_score(t_text, a_text):
    # Code based on SQUAD, compute_f1 
    t_tokens = [x for x in t_text]
    a_tokens = [x for x in a_text]
    common = collections.Counter(t_tokens) & collections.Counter(a_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(t_text)
    recall = 1.0 * num_same / len(a_text)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate(test_dict, answer_dict) -> Dict[str, float]:
    assert len(test_dict) == len(answer_dict)
    
    metrics = {'EM': 0, 'f1': 0}
    
    total = 0
    exact_match = 0
    f1 = 0

    for idx, t_line in enumerate(test_dict):
        test_context, answer_context = test_dict[t_line], answer_dict[t_line]
        if answer_context == "":
            continue

        total += 1
        
        test_processed = re.sub('\W+', '', test_context)
        answer_processed = re.sub('\W+', '', answer_context)

        exact_match += (test_processed == answer_processed)
        f1 += f1_score(test_context, answer_context)

    if total != 0:
        metrics['EM'] = exact_match / total
        metrics['f1'] = f1 / total
    else:
        print('validation 파일을 확인해주세요!')
        
    return metrics

def main(args):
    with open(args.test_file, encoding='utf-8') as f:
        test_file = json.load(f)
    
    with open(args.answer_file, encoding='utf-8') as f:
        answer_file = json.load(f)
    
    metrics = evaluate(test_file, answer_file)

    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # args
    parser.add_argument('--test_file', type=str, default="./outputs/test_dataset/predictions.json")
    parser.add_argument('--answer_file', type=str, default="./validation/validation_test.json")

    args = parser.parse_args()

    main(args)
