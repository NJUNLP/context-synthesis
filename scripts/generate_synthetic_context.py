import pdb
import json
import random
import re
import os
from tqdm import tqdm
from datasets import load_dataset
from multiprocessing import freeze_support
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
 
def load_dataset(dataset_name):
    dataset = []
    with open('../seed_instruction/{}.jsonl'.format(dataset_name), 'r') as fin:
        for line in fin:
            data = json.loads(line)
            dataset.append(data)
    
    return dataset

def call_api(prompt, model, tokenizer):
    messages=[
        {
            "role": "system", 
            "content": "Please infer the missing context. Always start with ``Context:'' and do not provide any explanation."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]

    messages = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    output = model.generate(messages, sampling_params)
    response = output[0].outputs[0].text

    return response

def encode_prompt(instruction_answer):
    instruction = instruction_answer['instruction']
    answer = instruction_answer['answer']

    prompt = "Context: [MISSING]\nQuestion: {}\nAnswer: {}\n\nThe above is a question-answer pair based on a context which is missing. Write the missing context to provide relevant background information that leads to both the question and the answer, ensuring that any necessary numerical or factual details are included. The context also should include relevant details about the character, their environment, aspirations, challenges, and relationships. It should be sufficiently detailed to reach approximately 2000 words.".format(instruction, answer)

    return prompt

def save_dataset(dataset, path):
    with open(path, 'w') as f:
        for data in dataset:
            json.dump(data, f)
            f.write('\n')

if __name__ == '__main__': 
    freeze_support()
    model = LLM(model="/cpfs01/shared/XNLP_H800/hf_hub/Qwen2.5-72B-Instruct",
            enforce_eager=True,
            max_num_batched_tokens=65536,
            tensor_parallel_size=8)
            
    sampling_params = SamplingParams(temperature=0.8, max_tokens=4096)

    tokenizer = AutoTokenizer.from_pretrained("/cpfs01/shared/XNLP_H800/hf_hub/Qwen2.5-72B-Instruct")

    random.seed(42)

    output_path = f"./all_synthesized_context.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing_lines = 0
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_lines = sum(1 for _ in f)

    with open(output_path, 'a') as f:
        processed_lines = 0

        for dataset_name in ['narrativeqa', 'qasper', 'hotpotqa', '2wikiqa', 'musique', 'govreport', 'qmsum', 'multinews']:
            dataset = load_dataset(dataset_name)
            pdb.set_trace()

            for i in tqdm(range(len(dataset))):
                processed_lines += 1
                if processed_lines <= existing_lines:
                    continue

                prompt = encode_prompt(dataset[i])

                try:
                    output = call_api(prompt, model, tokenizer)
                    context_pattern = r'Context:\s*(.*)'
                    context = re.findall(context_pattern, output, re.DOTALL)[0].strip() 

                    data = {"synthesis_task": dataset_name,
                            "instruction": dataset[i]['instruction'],
                            "answer": dataset[i]['answer'],
                            "synthesized_context": context}

                except IndexError:
                    print('Parsing fails')
                    context = output
                    data = {"synthesis_task": dataset_name,
                            "instruction": dataset[i]['instruction'],
                            "answer": dataset[i]['answer'],
                            "synthesized_context": context}

                json.dump(data, f)
                f.write('\n')