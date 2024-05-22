from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, GenerationConfig
from PIL import Image
import requests
import torch

model_id = "google/paligemma-3b-mix-448"
device = "cuda:0"
dtype = torch.bfloat16

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Instruct the model to create a caption in Spanish
prompt = "Does this figure show \"There are three doors\"? Please answer yes or no."
labels = ["yes"]
labels = processor(text=labels, images=image, return_tensors="pt").input_ids.to(model.device)
labels = [label[-2] for label in labels]
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]
with torch.inference_mode():
    cfg = GenerationConfig(
        output_logits=True, 
        max_new_tokens=2,
        return_dict_in_generate=True,
        do_sample=False,
    )
    logits = model(**model_inputs).logits[:, -1, :]
    lm_prob = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    for k in range(lm_prob.shape[0]):
        lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp() # exp to cancel the log and get raw prob between 0 and 1
    print("probabilities:", lm_prob)
