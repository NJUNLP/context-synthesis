#! /bin/bash

echo -e "This script can be used to synthesize the context via the OpenAI API,\nor other APIs that supports the OpenAI format and batched requests."

# Generate synthetic context
echo "Please set your API key in the two .py files below."
python request_synthetic_context_with_api.py

# Post-process the synthetic context after the batch job is completed
echo "Post-processing the synthetic context... Please make sure to provide a batch job ID."
python post_process_synthetic_context_with_api.py

# Convert to chat format
python convert_to_chat_format_with_api.py