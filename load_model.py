# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model = AutoModelForZeroShotImageClassification.from_pretrained("google/siglip-so400m-patch14-384")