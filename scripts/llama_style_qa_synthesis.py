
import json
import os
import sys
import time
import random
random.seed(42)
import argparse

from tqdm import tqdm
from functools import partial

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

SYSTEM_PROMPT = "Please create a question and its answer based on the background text given to you. Aways begin the question with 'Question:' and then begin the answer with 'Answer:'. Do not provide any explanation."

TEMPLATE = "Context:\n{context}\n\nThe above is a piece of text providing some background information. Write a question based on this context and then provide the corresponding answer. One must be able to infer the answer from the context information."


def read_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def openai_api(prompt, client, model, max_tokens=4096, max_attempt=5, wait_time=0.5):
    cur_attempt = 0
    while cur_attempt < max_attempt:
        time.sleep(wait_time)
        cur_attempt += 1
        try:
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, 
                          {"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
            )
            response_content = response.choices[0].message.content
            if "Question:" in response_content and "Answer:" in response_content:
                question = response_content.split("Question:")[-1].split("Answer:")[0].strip()
                answer = response_content.split("Answer:")[-1].strip()
                return question, answer
            else:
                print(response, flush=True)
                print("Could not parse the response. Retrying ...", flush=True)

        except Exception as e:
            print(e, flush=True)
            print(" Retrying ...", flush=True)
        
        return "Failed", "Failed"


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--save_filename', type=str, required=True)
    args.add_argument('--chunk_size', type=int, default=3000)
    args.add_argument('--model', type=str, default="gpt-4o-mini-2024-07-18")
    args.add_argument('--max_tokens', type=int, default=4096)
    args.add_argument('--max_attempts', type=int, default=5)
    args.add_argument('--wait_time', type=float, default=0.5)
    return args.parse_args()


if __name__ == '__main__':

    args = get_args()

    if os.path.exists(args.save_filename):
        print(f"File {args.save_filename} already exists. Please check.", flush=True)
        sys.exit(0)
    else:
        if args.model in ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18"]:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
            LLM = partial(openai_api, client=client, model=args.model, max_tokens=args.max_tokens, max_attempt=args.max_attempts, wait_time=args.wait_time)
        else:
            raise ValueError(f"Model {args.model} is not implemented or supported.")

    
    with open(args.save_filename, "w") as f_out:
        data_names = ['narrativeqa', 'qasper', 'hotpotqa', '2wikiqa', 'musique', 'govreport', 'qmsum', 'multinews']

        for data_name in data_names:
            data = read_jsonl(f"../seed_context/{data_name}.jsonl")
        
            for d in tqdm(data, desc=f"Synthesizing questions and answers for {data_name} based on the input context..."):
                full_context = d["context"]
                words = full_context.split()
                full_context_length = len(words)
                if full_context_length < args.chunk_size:
                    context = full_context
                else: # randomly sample args.chunk_size continuous space-separated words from the full context

                    # first, decide a random start position
                    start_word = random.randint(0, full_context_length - args.chunk_size - 1)
                    
                    # then, find the start position
                    start_pos = 0
                    for i in range(start_word):
                        start_pos += len(words[i])
                        while start_pos < len(full_context) and full_context[start_pos].isspace():
                            start_pos += 1
                    
                    # then, find the end position
                    end_word = start_word + args.chunk_size
                    end_pos = start_pos
                    for i in range(start_word, end_word):
                        end_pos += len(words[i])
                        while end_pos < len(full_context) and full_context[end_pos].isspace():
                            end_pos += 1
                    
                    # finally, extract the context
                    context = full_context[start_pos:end_pos]

                prompt = TEMPLATE.format(context=context)
                instruction, answer = LLM(prompt=prompt)
                
                # include generated instruction and answer that are needed for training
                d["instruction"] = instruction
                d["answer"] = answer
                
                # some meta-data about the task name, sampled context, and model name
                d["task"] = data_name
                d["sampled_context"] = context
                d["model"] = args.model

                f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
        