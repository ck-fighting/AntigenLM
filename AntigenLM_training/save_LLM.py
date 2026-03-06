from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel

# 1. 加载 base 模型 + LoRA adapter
base_model = AutoModelForMaskedLM.from_pretrained("/data0/chenkai/data/microLM-main/Result_PathogLM_300M_SS")
tokenizer = AutoTokenizer.from_pretrained("/data0/chenkai/data/microLM-main/Result_PathogLM_300M_SS")
lora_model = PeftModel.from_pretrained(base_model, "/data0/chenkai/data/microLM-main/Result_finetune_LoRA_pathogLM_300M_SS/checkpoint-34900")

# 2. 合并 LoRA 到 base 模型
merged_model = lora_model.merge_and_unload()

# 3. 保存为完整的 PathogLM
save_dir = "/data0/chenkai/data/microLM-main/Result_antigenLM_300M_SS_2"
merged_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
