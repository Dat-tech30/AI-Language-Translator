import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "decapoda-research/llama-3.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "What is the capital of France?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

generated_text = model.generate(input_ids, max_length=100, num_beams=4)