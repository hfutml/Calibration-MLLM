import argparse
import json
import os
import torch
import logging
import numpy as np
from PIL import Image
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score

# 默认设备设置

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def setup_model(model_path, tokenizer_path, device):
    """Load model and tokenizer."""
    logging.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def is_image_normal(file_path):
    """Check if image file is valid."""
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        logging.warning(f"File {file_path} is empty or does not exist.")
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify if image is corrupted
            return True
    except (IOError, SyntaxError) as e:
        logging.error(f"Error opening image {file_path}: {e}")
        return False

def create_one_hot(n_classes, label_smoothing=0):
    """Create one-hot encoded tensor."""
    smoothing_value = label_smoothing / (n_classes - 1)
    one_hot = torch.full((n_classes,), smoothing_value).float()
    return one_hot

def cross_entropy(output, target, n_classes, label_smoothing=0):
    """Compute cross-entropy with label smoothing."""
    model_prob = create_one_hot(n_classes, label_smoothing)
    model_prob[target] = 1. - label_smoothing
    return F.kl_div(output, model_prob, reduction='sum').item()

def get_confidence_answer(prompt, image_path, model, tokenizer,temperature=1, add_image=True, max_steps=20):
    """Get the confidence and predicted answer."""
    with torch.no_grad():
        if add_image:
            query = tokenizer.from_list_format([{'image': image_path}, {'text': prompt}])
        else:
            query = tokenizer.from_list_format([{'text': prompt}])

        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        input_ids = inputs['input_ids'].to(model.device)
        token_sequence = input_ids

        for step in range(max_steps):
            outputs = model(token_sequence)
            logits = outputs.logits[:, -1, :]/temperature
            probabilities = F.softmax(logits, dim=-1)
            max_confidence, max_token_id = torch.max(probabilities, dim=-1)

            decoded_token = tokenizer.decode(max_token_id.item(), skip_special_tokens=True)

            if decoded_token in ['A', 'B', 'C', 'D', 'E']:
                return decoded_token, max_confidence.item(), logits

            token_sequence = torch.cat([token_sequence, max_token_id.unsqueeze(0)], dim=1)

        return 'N', 0, None

def calculate_ece(data):
    """Calculate Expected Calibration Error (ECE)."""
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

def ece_tes(args,model,tokenizer,temperature=1):
    with open(args.test_data_path, 'r') as file:
        data = json.load(file)
    k, sum, list_log, list_bin = 0, 0, [], []
    for i in range(10):
        list_bin.append({'id': i, 'num': 0, 'k': 0, 'cof_sum': 0})

    for item in tqdm(data, desc="Testing model"):
        prompt = item['conversations'][0]['value'] + "\nPlease provide the option letter directly."
        answer = item['conversations'][1]['value']
        image_id = item['image']
        image_path = os.path.join(args.image_folder, image_id)

        if not is_image_normal(image_path):
            continue

        prompt = prompt + "\nAnswer: ("
        answer_pre, confidence, _ = get_confidence_answer(prompt, image_path, model, tokenizer,temperature, max_steps=20)

        if answer_pre not in ['A', 'B', 'C', 'D', 'E']:
            continue

        sum += 1
        list_log.append(confidence)
        list_bin[min(int(confidence * 10), 9)]['num'] += 1
        list_bin[min(int(confidence * 10), 9)]['cof_sum'] += confidence
        if answer == answer_pre:
            k += 1
            list_bin[min(int(confidence * 10), 9)]['k'] += 1

    ece_value = calculate_ece(list_bin)
    print(f"ECE: {ece_value:.4f}")


def main(args):
    # Set up model and tokenizer
    model, tokenizer = setup_model(args.model_path, args.tokenizer_path, args.device)

    print("ECE testing before calibration of the test set.....")
    ece_tes(args,model,tokenizer)

    # Load training data
    with open(args.train_data_path, 'r') as file:
        data = json.load(file)

    list_res = []
    for item in tqdm(data, desc="Processing training data"):
        prompt = item['conversations'][0]['value'] + "\nPlease provide the option letter directly."
        answer = item['conversations'][1]['value']
        image_id = item['image']
        image_path = os.path.join(args.image_folder, image_id)

        if not is_image_normal(image_path):
            continue

        prompt = prompt + "\nAnswer: ("
        answer_pre, confidence, logits = get_confidence_answer(prompt, image_path, model, tokenizer, add_image=False, max_steps=20)

        if answer_pre not in ['A', 'B', 'C', 'D', 'E']:
            continue
   
        dict4ts = {
            'logits': logits.squeeze().cpu(),
            'pred': [tokenizer.encode(answer_pre, add_special_tokens=False)[0]],
            'true': [tokenizer.encode(answer, add_special_tokens=False)[0]]
        }
        list_res.append(dict4ts)

    # Train temperature scaling
    n_classes = len(list_res[0]['logits'])

    best_nll = float('inf')
    best_temperature = -1

    for temp in tqdm(range(1000), desc="Training temperature scaling"):
        temp_value = round(temp / 100, 10)
        nll = np.mean([cross_entropy(F.log_softmax(elem['logits'] / temp_value, 0), elem['true'], n_classes) for elem in list_res])
        if nll < best_nll:
            best_nll = nll
            best_temperature = temp_value

    temperature = best_temperature
    logging.info(f"Best temperature: {temperature}")


    print("ECE testing after calibration of the test set.....")
    ece_tes(args,model,tokenizer,temperature=temperature)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature scaling and evaluation")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer")
    parser.add_argument('--device', type=str, required=True, help="cuda:0")
    parser.add_argument('--train_data_path', type=str, required=True, help="Path to the training data")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the test data")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the image folder")

    args = parser.parse_args()

    main(args)
