import json
import os
import re
import string
from typing import List, Dict
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import parse, verify
import argparse

def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)

    if grade_answer(answer, ground_truth):
        reward = 1.0
    elif float(verify(parse(answer), parse(ground_truth))) > 0:
        reward = 1.0
    elif (
        len(ground_truth) == 1
        and ground_truth in string.ascii_uppercase
        and ground_truth.upper() in answer.upper()
    ):
        reward = 1.0
    else:
        reward = 0.0

    return reward

def compute_score(results: List[Dict]) -> float:
    total_score = 0.0
    total_count = len(results)
    
    for result in results:
        predict = result['prediction']
        ground_truth = result['answer']
        accuracy_score = accuracy_reward(predict, ground_truth)
        result['score'] = accuracy_score  

        total_score += accuracy_score

    accuracy = total_score / total_count * 100 if total_count > 0 else 0.0

    return round(accuracy, 2)

def process_json_files(folder_path: str) -> None:
    target_files = {
        'mathverse.json',
        'hallubench.json',
        'mathvista.json',
        'wemath.json',
        'mathvision.json',
        'GeoMath.json',
        'Tallyqa.json',
        'MME.json'
    }

    for filename in os.listdir(folder_path):
        if filename in target_files: 
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    accuracy = compute_score(data['results'])
                    # accuracy = compute_score(data)
                    
                    data['config'] = data.get('config', {})
                    data['config']['accuracy'] = accuracy

                    new_filename = filename.replace('.json', '_score.json')
                    new_file_path = os.path.join(folder_path, new_filename)
                    with open(new_file_path, 'w', encoding='utf-8') as new_file:
                        json.dump(data, new_file, ensure_ascii=False, indent=4)
                    
                    print(f'{filename}: {accuracy:.2f}')
                except json.JSONDecodeError:
                    print(f'Unable to parse file: {filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and compute accuracy.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing JSON result files.")
    args = parser.parse_args()
    process_json_files(args.folder_path)