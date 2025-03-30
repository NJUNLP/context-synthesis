import json
import copy

from openai import OpenAI

OPENAI_API_KEY=""
assert OPENAI_API_KEY, "The API key is not correctly set."

# the batch ID of the OpenAI batch job
BATCH_ID = ""
assert BATCH_ID, "The batch ID is not correctly set."


TEMPLATE_REQUEST = {
    "custom_id": "",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [
            {"role": "system", "content": """Please infer the missing context. Always start with "Context:" and do not provide any explanation."""},
            {"role": "user", "content": ""}
        ],
        "max_completion_tokens": 4096
    }
}


def read_jsonl_data(filename, synthesis_task):
    data = []
    with open(filename, "r") as f:
        for line in f:
            line_data = json.loads(line.strip())
            line_data["synthesis_task"] = synthesis_task
            data.append(line_data)

    return data


def template_question_answer(data:dict, length:int=2000):

    assert isinstance(data, dict) and "instruction" in data and "answer" in data
    question, answer = data["instruction"], data["answer"]

    s = f"Context: [MISSING]\nQuestion: {question}\nAnswer: {answer}\n\n"
    s += f"The above is a question-answer pair based on a context which is missing. Write the missing context to provide relevant background information that leads to both the question and the answer, ensuring that any necessary numerical or factual details are included. The context also should include relevant details about the character, their environment, aspirations, challenges, and relationships. It should be sufficiently detailed to reach approximately {length} words."

    return s


if __name__ == "__main__":
    # read the seed instruction-answer pairs from the eight files
    all_data = []
    for task_name in ['qmsum', 'multinews', 'govreport', 'narrativeqa', 'qasper', 'hotpotqa', '2wikiqa', 'musique']:
        all_data += read_jsonl_data(f"./../seed_instruction/{task_name}.jsonl", synthesis_task=task_name)

    # template the seed question-answer pairs
    all_request_data = [template_question_answer(d) for d in all_data]

    # create the request file
    with open(f"batch_request_context_synthesis_with_api.jsonl", "w") as f_batch_request:
        for i, (data, metadata) in enumerate(zip(all_request_data, all_data)):
            request = copy.deepcopy(TEMPLATE_REQUEST)
            request["custom_id"] = f"{metadata['synthesis_task']}_{i}"
            request["body"]["messages"][1]["content"] = data

            f_batch_request.write(json.dumps(request, ensure_ascii=False) + "\n")

    # initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # create (upload) the batch input file
    batch_input_file = client.files.create(
        file=open(f"batch_request_context_synthesis_with_api.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    print(f"Input File ID: {batch_input_file_id}")

    # create (start) the batch job
    returned = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Context synthesis requests for longbench."
        }
    )
    batch_id = returned.id
    print(f"Batch ID: {batch_id}")
    print(returned)
