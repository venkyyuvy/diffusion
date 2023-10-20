import torch

# For video display:
from transformers import CLIPTextModel, CLIPTokenizer, logging, \
 CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import torch.nn.functional as F

import os

torch.manual_seed(1)


# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Load the tokenizer and text encoder to tokenize and encode the text.
clip_model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(torch_device);
vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(torch_device);
processor = CLIPProcessor.from_pretrained(clip_model_name)
# additional textual prompt
textual_direction = "on a beach"
inputs = processor(text=textual_direction,
                   return_tensors="pt",
                   padding=True)
with torch.no_grad():
    text_embed = CLIPTextModelWithProjection.from_pretrained(
        clip_model_name)(**inputs).text_embeds.to(torch_device)

def cosine_loss(gen_image, text_embed=text_embed):

    gen_image_clamped = gen_image.clamp(0, 1).mul(255)
    resized_image = F.interpolate(gen_image_clamped,
                                  size=(224, 224),
                                  mode='bilinear',
                                  align_corners=False)
    image_embed = vision_encoder(resized_image).image_embeds
    similarity = F.cosine_similarity(text_embed, image_embed, dim=1)
    loss = 1 - similarity.mean()
    return loss


