import json
import os

from openai import OpenAI

OPENAI_API_KEY=""
assert OPENAI_API_KEY, "The API key is not correctly set."

# the batch ID of the OpenAI batch job
BATCH_ID = ""
assert BATCH_ID, "The batch ID is not correctly set."


if __name__ == "__main__":

    # initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # retrieve the batch job
    retrieved = client.batches.retrieve(BATCH_ID)

    if not os.path.exists(f"batch_output_context_synthesis_with_api.jsonl"):
        if retrieved.status == "completed" and retrieved.output_file_id is not None:
            file_data = client.files.content(retrieved.output_file_id)
            file_data_bytes = file_data.read()
            with open(f"batch_output_context_synthesis_with_api.jsonl", "wb") as file:
                file.write(file_data_bytes)
        else:
            exit(f"Batch {BATCH_ID} has not completed.")
    else:
        print(f"Batch {BATCH_ID}'s output file already exists.")

    # read the batch output file to post-process for the synthesized context
    synthetic_context_data = []
    with open(f"batch_output_context_synthesis_with_api.jsonl", "r") as f:
        for line in f:
            synthetic_context_data.append(json.loads(line.strip()))

    # read the seed instruction data from the eight files
    seed_data = []
    for task_name in ['qmsum', 'multinews', 'govreport', 'narrativeqa', 'qasper', 'hotpotqa', '2wikiqa', 'musique']:
        seed_data += read_jsonl_data(f"seed_instruction/{task_name}.jsonl", synthesis_task=task_name)

    assert len(synthetic_context_data) == len(seed_data)

    processed_synthetic_context_data = []
    for synthesis_instance, seed_instance in zip(synthetic_context_data, seed_data):
        synthesis_task = seed_instance["task"]
        context_raw = synthesis_instance["response"]["body"]["choices"][0]["message"]["content"]
        context = context_raw.split("Context:", 1)[1].strip()

        processed_synthetic_context_data.append({
            "synthesis_task": synthesis_task,
            "synthesized_context": context,
            "instruction": seed_instance["instruction"],
            "answer": seed_instance["answer"],
        })

    with open(f"all_synthesized_context.jsonl", "w") as f:
        for data in processed_synthetic_context_data:
            f.write(json.dumps(data) + "\n")
