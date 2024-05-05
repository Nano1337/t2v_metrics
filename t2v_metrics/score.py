from abc import abstractmethod
from typing import List, TypedDict, Union
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .constants import HF_CACHE_DIR

class ImageTextDict(TypedDict):
    images: List[str]
    texts: List[str]

class Score(nn.Module):

    def __init__(self,
                 model: str,
                 device: str='cuda',
                 cache_dir: str=HF_CACHE_DIR):
        """Initialize the ScoreModel
        """
        super().__init__()
        assert model in self.list_all_models()
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir)
    
    @abstractmethod
    def prepare_scoremodel(self,
                           model: str,
                           device: str,
                           cache_dir: str):
        """Prepare the ScoreModel
        """
        pass
    
    @abstractmethod
    def list_all_models(self) -> List[str]:
        """List all available models
        """
        pass

    def forward(self,
                images: Union[str, List[str]],
                texts: Union[str, List[str]],
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        if type(images) == str:
            images = [images]
        if type(texts) == str:
            texts = [texts]
        
        scores = torch.zeros(len(images), len(texts)).to(self.device)
        for i, image in enumerate(images):
            scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
        return scores
    
    def batch_forward(self,
                      dataset: List[ImageTextDict],
                      batch_size: int=16,
                      **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        num_samples = len(dataset)
        num_images = len(dataset[0]['images'])
        num_texts = len(dataset[0]['texts'])
        scores = torch.zeros(num_samples, num_images, num_texts).to(self.device)
        
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        counter = 0
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            cur_batch_size = len(batch['images'][0])
            assert len(batch['images']) == num_images, \
                f"Number of image options in batch {batch_idx} is {len(batch['images'])}. Expected {num_images} images."
            assert len(batch['texts']) == num_texts, \
                f"Number of text options in batch {batch_idx} is {len(batch['texts'])}. Expected {num_texts} texts."
            
            for image_idx in range(num_images):
                images = batch['images'][image_idx]
                for text_idx in range(num_texts):
                    texts = batch['texts'][text_idx]
                    out = self.model.forward(images, texts, **kwargs)
                    scores[counter:counter+cur_batch_size, image_idx, text_idx] = out
            
            counter += cur_batch_size
        return scores
    
"""

    for tag in tags:
        tag_result[tag] = {}
        mscore, hscore = [], []
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr

        # Plot our scores subset for all models
        plt.figure(figsize=(10, 6))
        for model in items_by_model_tag[tag]:
            model_indices = items_by_model_tag[tag][model]
            our_scores_subset = our_scores[model_indices].flatten()
            human_scores_subset = np.array(human_scores)[model_indices]
            plt.scatter(human_scores_subset, our_scores_subset, label=model)

        # Assume points are paired. Redo scatter plot
        plt.plot(human_scores_subset, our_scores_subset, 'ro')

        # Calculate and plot correlation score
        r, _ = pearsonr(human_scores_subset, our_scores_subset)
        plt.title(f'Correlation Score: {r:.2f}')
        plt.legend()
        plt.savefig(f'{tag}_scores.png')
        plt.close()
        exit()
"""