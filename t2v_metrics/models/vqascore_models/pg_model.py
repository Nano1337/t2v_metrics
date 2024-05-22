from typing import List
import torch
import copy
import os
from torchvision import transforms
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from .vqa_model import VQAScoreModel
from .lavis.models import load_model
from ...constants import HF_CACHE_DIR


default_question_template = 'Question: Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "yes" # paligemma uses "yes" instead of "Yes"

PALIGEMMA_MODELS = {
    'paligemma-3b-mix-448': {'variant': '3b-mix-448'},
    # TODO: benchmark more models, should be simple
}

class PaliGemmaModel(VQAScoreModel): 
    """A wrapper for the PaliGemma models"""
    def __init__(self, 
                 model_name='paligemma-3b-mix-448',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in PALIGEMMA_MODELS, f"Model name {model_name} not found in PALIGEMMA_MODELS"
        os.environ['TORCH_HOME'] = cache_dir
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self): 
        """
        Load the model, tokenizer, image transform
        """
        self.variant = PALIGEMMA_MODELS[self.model_name]['variant']
        name = "google/paligemma-" + self.variant
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            revision="bfloat16",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(name)

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """
        Load the image(s), further processing done later
        """
        return [self.image_loader(x) for x in image]

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str=default_question_template,
                answer_template: str=default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Is the image showing 'a photo of a dog'? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        images = self.load_images(images)

        labels = self.processor(text=answers, images=images, return_tensors="pt").input_ids.to(self.device)
        labels = [label[-2] for label in labels]    

        model_inputs = self.processor(text=questions, images=images, return_tensors="pt", padding=True).to(self.device)

        # get model predictions
        logits = self.model(**model_inputs).logits[:, -1, :]

        # calculate probability of each label given model output
        lm_prob = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp() # exp to cancel the log and get raw prob between 0 and 1
    
        return lm_prob


