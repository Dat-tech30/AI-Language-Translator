# First install the "pip install transformers torch"
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "decapoda-research/llama-3.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_text = "What is the capital of France?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

generated_text = model.generate(input_ids, max_length=100, num_beams=4)
generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(generated_text)