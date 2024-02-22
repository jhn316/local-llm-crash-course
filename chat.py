from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

"""
import torch
import transformers

branch = "200B"
llm = transformers.AutoModelForCausalLM.from_pretrained(
    "LumiOpen/Poro-34B",
    torch_dtype=torch.bfloat16,
    revision=branch,
)
"""

prompt = "What words complete the sentence The capital of India is"

# print(llm(prompt))
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
