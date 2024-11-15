from transformers import AutoTokenizer, LlamaForCausalLM

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

model = LlamaForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prompt = '''
I am
'''
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {
    "input_ids": inputs.input_ids,
    "attention_mask": inputs.attention_mask,
}

print(inputs)

output = model(**inputs)

print(output)

# Generate
# generate_ids = model.generate(inputs.input_ids, max_length=10)

# print(generate_ids)

# outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print(outputs[0])