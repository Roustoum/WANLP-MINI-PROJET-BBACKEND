#coding:UTF-8

from transformers import BartTokenizer, BartForConditionalGeneration
import torch


tokenizer = BartTokenizer.from_pretrained('./CohereZiGenerator')
model = BartForConditionalGeneration.from_pretrained('./CohereZiGenerator')

device = torch.device("cpu")
model.to(device)

prompt = "9law el jaj fel el kabouya"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=20, do_sample=True)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)