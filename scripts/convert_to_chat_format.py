import pdb
import json
import random
from collections import defaultdict

random.seed(42)

dataset2prompt = {"narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
                  "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
                  "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
                  "2wikiqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
                  "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
                  "govreport": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
                  "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
                  "multinews": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
                  }

dataset_names = ['narrativeqa', 'qasper', 'hotpotqa', '2wikiqa', 'musique', 'govreport', 'qmsum', 'multinews']
path = "./all_synthesized_context.jsonl"

content_set = set()
dataset2list = defaultdict(list)

with open(path, "r") as f:
    for line in f:
        try:
            sample = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(sample)

        dataset = sample["synthesis_task"]
        instruction = sample['instruction']
        answer = sample['answer']
        context = sample['synthesized_context']
        
        data = (instruction, answer, context)

        dataset2list[dataset].append(data)

print([len(dataset2list[key]) for key in dataset2list.keys()])

IRR_CONTEXT_NUM = 9
saved_dataset = []
for dataset in ['qmsum', 'multinews', 'govreport', 'narrativeqa', 'qasper', 'hotpotqa', '2wikiqa', 'musique']:
    for i in range(len(dataset2list[dataset])):
        rel_data = dataset2list[dataset][i]

        irr_datas = set()
        while len(irr_datas) < IRR_CONTEXT_NUM:
            irr_data = random.choice(dataset2list[dataset])
            if irr_data not in irr_datas and irr_data != rel_data:
                irr_datas.add(irr_data)
        
        rel_question = rel_data[0]
        rel_answer = rel_data[1]
        rel_context = rel_data[2]
        irr_contexts = [irr_data[2] for irr_data in irr_datas]

        contexts = [rel_context] + irr_contexts
            
        random.shuffle(contexts)

        question = rel_question
        answer = rel_answer
        context = '\n\n'.join(contexts)

        user_content = dataset2prompt[dataset].format(context=context, input=question)
        assistant_content = answer 

        data = {"messages": [{"role": "user", "content": user_content},
                             {"role": "assistant", "content": assistant_content}]}

        saved_dataset.append(data)

save_path = "./context-synthesis-chat-format.jsonl"

pdb.set_trace()
with open(save_path, 'w') as f:
    for data in saved_dataset:
        json.dump(data, f)
        f.write('\n')