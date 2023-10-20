import torch

import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import CLIPTextModel, CLIPTokenizer, \
 CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

import os
# from image_generator import get_output_embeds, position_embeddings


# Set device
torch_device = "cuda" if torch.cuda.is_available() else "mps" \
    if torch.backends.mps.is_available() else "cpu"

if "mps" == torch_device: 
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Load the tokenizer and text encoder to tokenize and encode the text.
clip_model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(torch_device);
vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(torch_device);
processor = CLIPProcessor.from_pretrained(clip_model_name)

# # additional textual prompt
def get_text_embed(prompt = "on a mountain"):
    inputs = processor(text=prompt,
                       return_tensors="pt",
                       padding=True)
    with torch.no_grad():
        text_embed = CLIPTextModelWithProjection.from_pretrained(
            clip_model_name)(**inputs).text_embeds.to(torch_device)
    return text_embed

# def get_text_embed(prompt = "on a mountain"):
#     text_input = tokenizer([prompt],
#                            padding="max_length",
#                            max_length=tokenizer.model_max_length,
#                            truncation=True,
#                            return_tensors="pt")
#     with torch.no_grad():
#         text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
#         input_embeddings = text_embeddings + position_embeddings.to(torch_device)
#     modified_output_embeddings = get_output_embeds(input_embeddings)
#     return modified_output_embeddings

text_embed = get_text_embed()
def cosine_loss(gen_image, text_embed=text_embed):
    gen_image_clamped = gen_image.clamp(0, 1).mul(255)
    resized_image = v2.Resize(224)(gen_image_clamped)
    image_embed = vision_encoder(resized_image).image_embeds
    similarity = F.cosine_similarity(text_embed, image_embed, dim=1)
    loss = 1 - similarity.mean()
    return loss

def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    return error
