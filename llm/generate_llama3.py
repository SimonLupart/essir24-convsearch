import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

login(token="[your hf token]")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, padding_side="left", maximum_length = 4096, model_max_length = 4096)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', quantization_config=bnb_config)

tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id

prompt= (["Instruction:\nGenerate a conversation between an user and an assistant about Amsterdam.\nConversation:\n"])

tokens = tokenizer(prompt, return_tensors="pt", truncation=True).to(0)
outputs = model.generate(input_ids=tokens["input_ids"], attention_mask= tokens["attention_mask"], max_new_tokens = 128, do_sample=True, eos_token_id= tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)
results = tokenizer.decode(outputs[0], skip_special_tokens=True)
# temperature=0.7,
print(results)


# with open("/projects/0/prjs0871/hackathon/conv-search/llm/generated/queries_rowid_dev_llm.tsv", "w") as tsv_queries_rw:
#     with open("/projects/0/prjs0871/hackathon/conv-search/DATA/queries_rowid_dev_all.tsv") as q_tsv:
#         for conv_sample in q_tsv:
#             conv_turn_id, conv = conv_sample.strip().split("\t")

#             prompt = (["Instruction:\nRewrite the last query from the conversation. Do not generate anything more\nConversation:\n"+conv+"\nRewrite:\n"])

#             tokens = tokenizer(prompt, return_tensors="pt", truncation=True).to(0)
#             outputs = model.generate(input_ids=tokens["input_ids"], attention_mask= tokens["attention_mask"], max_new_tokens = 128, do_sample=True, eos_token_id= tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)
#             results = tokenizer.decode(outputs[0], skip_special_tokens=True)

#             rewrite = results.split("\nRewrite:\n")[1]
#             tsv_queries_rw.write(conv_turn_id+"\t"+rewrite+"\n")

