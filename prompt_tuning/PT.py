import argparse
import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import time
from openai import OpenAI, OpenAIError
import numpy as np
from PIL import Image
import re
from openai import OpenAI

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="AI Model Inference Script")
    parser.add_argument('--json_path', required=True, help="Path to the JSON file with data")
    parser.add_argument('--image_fold_path', required=True, help="Path to the folder containing images")
    parser.add_argument('--model_path', required=True, help="Path to the model directory")
    parser.add_argument('--logging_path', required=True, help="Path to the logging file")
    parser.add_argument('--API_key', required=True, help="OpenAI API key")
    parser.add_argument('--device', default='cuda:7', help="Device to use for computation (e.g., 'cuda:0', 'cpu')")
    parser.add_argument('--k', type=int, default=5, help="Top-k prompts to retain after each iteration")
    parser.add_argument('--m', type=int, default=5, help="Number of synonyms to generate for each prompt")
    parser.add_argument('--n', type=int, default=10, help="Number of generations/iterations for the algorithm")
    parser.add_argument('--seed_prompts', nargs='+', default=["The answer might be", "The answer must be"], help="Seed prompts for the algorithm")
    return parser.parse_args()

# Configure logger
def setup_logger(logging_path):
    logging.basicConfig(filename=logging_path, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def chatGPT(prompt, api_key,temperature=0.7, top_p=0.95, max_retries=5, retry_delay=2):
    client = OpenAI(api_key=api_key, base_url="base_url")
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            return answer

        except OpenAIError as e:
            print(f"Request failed: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying ({retries}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Could not complete request.")
                return None

# Generate synonyms using GPT model
def generate_synonyms(suffix, m, api_key, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            prompt = f"Given sentence: \"{suffix}\", please generate {m} sentences similar to the given sentence, separate each sentence with '<sx>', be careful not to output anything except for the sentence."
            response = chatGPT(prompt, api_key)

            synonyms = [re.sub(r'^\d+\.\s*', '', synonym.strip().rstrip('.').strip('"')) for synonym in response.split("<sx>")]
            return synonyms
        except Exception as e:
            retries += 1
            logging.error(f"Attempt {retries}/{max_retries} failed: {e}")
    logging.warning("Max retries reached, returning empty list.")
    return []


def group_by_accuracy(accuracy, interval=0.2):
    return int(accuracy // interval)


def ape_algorithm(seed_prompts, eval_function, k, m, n, api_key, json_path, image_fold_path, model, tokenizer):
    current_generation = seed_prompts
    best_prompt = None
    best_score = float('inf')
    best_accuracy = 0

    for i in tqdm(range(n), desc="Progress"):
        next_generation = []
        for prompt in current_generation:
            next_generation.extend(generate_synonyms(prompt, m, api_key))

        current_generation = list(set(current_generation + next_generation))
        scored_prompts = [(prompt, *eval_function(prompt, json_path, image_fold_path, model, tokenizer)) for prompt in current_generation]
        scored_prompts.sort(key=lambda x: (-group_by_accuracy(x[2]), x[1]))

        current_generation = [prompt for prompt, _, _ in scored_prompts[:k]]

        if scored_prompts[0][2] > best_accuracy or (scored_prompts[0][2] == best_accuracy and scored_prompts[0][1] < best_score):
            best_prompt = scored_prompts[0][0]
            best_score = scored_prompts[0][1]
            best_accuracy = scored_prompts[0][2]

        logging.info(f"Round {i+1}: Best prompt: '{best_prompt}', Score: {best_score}, Accuracy: {best_accuracy}")
        logging.info("Top k prompts this round:")
        for j, (prompt, score, accuracy) in enumerate(scored_prompts[:k], start=1):
            logging.info(f"{j}. Prompt: '{prompt}', Score: {score}, Accuracy: {accuracy}")

    return best_prompt, best_score, best_accuracy

def chat_inference(prompt, image_path, model, tokenizer, add_image=True,max_steps=20):
    with torch.no_grad():
        # Prepare the input query
        if add_image:
            query = tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt},
            ])
        else:
            query = tokenizer.from_list_format([
                {'text': prompt},
            ])

        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        input_ids = inputs['input_ids'].to(model.device)
        token_sequence = input_ids


        for step in range(max_steps):
            outputs = model(token_sequence)
            logits = outputs.logits[:, -1, :]  
 
            probabilities = F.softmax(logits, dim=-1)
            max_confidence, max_token_id = torch.max(probabilities, dim=-1)

            decoded_token = tokenizer.decode(max_token_id.item(), skip_special_tokens=True)

            if decoded_token in ['A', 'B', 'C', 'D', 'E']:
                return max_confidence.item(), decoded_token

            token_sequence = torch.cat([token_sequence, max_token_id.unsqueeze(0)], dim=1)

        return 0, 'N'


# Calculate ECE
def calculate_ece(data):
    total_samples = sum(d["num"] for d in data)
    ece = 0.0
    for d in data:
        if d['num'] == 0:
            continue
        acc = d['k'] / d['num']
        avg_conf = d['cof_sum'] / d['num']
        num_samples = d["num"]
        ece += (np.abs(acc - avg_conf) * num_samples) / total_samples
    return ece

# Check if image is normal (valid)
def is_image_normal(file_path):
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        logging.warning(f"File is empty or does not exist: {file_path}")
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verifies if the image is valid
            return True
    except (IOError, SyntaxError) as e:
        logging.error(f"Error opening image: {e}")
        return False

# Evaluation function
def eval_function(suffix, json_path, image_fold_path, model, tokenizer):
    with open(json_path, 'r') as file:
        data = json.load(file)

    k = 0
    cof_sum = 0
    total = 0
    list_log = []
    list_bin = [{'id': i, 'num': 0, 'k': 0, 'cof_sum': 0} for i in range(10)]
    err_times = 0
    logging.info('Testing.....')
    for item in data:
        prompt = item['conversations'][0]['value']
        prompt+="Please provide the option letter directly."
        answer = item['conversations'][1]['value']
        image_id = item['image']

        prompt = f"{prompt}\n{suffix}: ("
        image_path = os.path.join(image_fold_path, image_id)
        image_path="no"

        if not is_image_normal(image_path):
            continue

        if err_times > 20:
            return (10000, 0)

        try:
            confidence, answer_forward = chat_inference(prompt, image_path, model, tokenizer, add_image=False)
            if answer_forward=="N":
                continue
    
        except Exception as e:
            err_times += 1
            logging.error(f"Error during inference: {e}")
            continue


        total += 1
        cof_sum += confidence
        list_log.append(confidence)

        list_bin[min(int(confidence * 10), 9)]['num'] += 1
        list_bin[min(int(confidence * 10), 9)]['cof_sum'] += confidence
        if answer[0] == answer_forward:
            k += 1
            list_bin[min(int(confidence * 10), 9)]['k'] += 1

    if total == 0:
        return (10000, 0)

    ece_value = calculate_ece(list_bin)
    return (ece_value, k / total)

# Main function to run the script
def main():
    args = parse_args()
    setup_logger(args.logging_path)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    best_prompt, best_score, _ = ape_algorithm(args.seed_prompts, eval_function, args.k, args.m, args.n, args.API_key, args.json_path, args.image_fold_path, model, tokenizer)
    
    logging.info(f"Best prompt: {best_prompt}")
    logging.info(f"Best score: {best_score}")

if __name__ == "__main__":
    main()
