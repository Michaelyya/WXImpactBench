from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig
import os
from huggingface_hub import login

from dotenv import load_dotenv

load_dotenv()

login(token="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")
# Ensure the API key is loaded
access_token = os.getenv("HUGGINGFACE_API_KEY")
if not access_token:
    raise ValueError("HUGGINGFACE_API_KEY is not set in environment variables.")

# Initialize the pipeline for text generation using the API token
pipe = pipeline("text-generation", model="meta-llama/Llama-2-13b-hf")

# Load configurations and model with PEFT enhancements
config = PeftConfig.from_pretrained("pykale/llama-2-13b-ocr")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
model = PeftModel.from_pretrained(base_model, "pykale/llama-2-13b-ocr")

# Configure quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model with specific quantization and performance configurations
model = AutoPeftModelForCausalLM.from_pretrained(
    'pykale/llama-2-13b-ocr',
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    use_auth_token=access_token
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('pykale/llama-2-13b-ocr', use_auth_token=access_token)

# Define the input text for OCR correction
ocr = """TURKEY', ' Bwsaltus snnwItloMs mt war', ' BrcnADHT', 'June 22, Russia has presented to Bulgaria another war-ship, also 16,000 rifles', ', Al baaia and rientenejrro', ' CossTAjmyoPL', ' June 22', ' The Porte de clines to force the Albanians to surrender their territory to Montenegro, but is willing to cse its persuasion', ' ArSTRIA-nTXGART', ' ', ' Klnteterial crisis', ' ', ' Yuxxa, June 22', ' A Ministerial crisis is imminent GEKJIAJY', ' ttiaaatrsnsi raiasu B kr Us', ' June 22', ' In the district of Lam- bar, in Breslau, Prussia, heavy torrential rains have killed 36 persons and destroyed 105 houses"""

prompt = f"""### Instruction:
Fix the OCR errors in the provided text.

### Input:
{ocr}

### Response:
"""

# Generate predictions
input_ids = tokenizer(prompt, max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()

# Output the corrected text
print(pred)
