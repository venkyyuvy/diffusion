from utils import latents_to_pil

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login

# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPVisionModel,\
 CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

import os

torch.manual_seed(1)
if not (Path.home()/'.cache/huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Prep Scheduler
def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32) 
    # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae").to(torch_device)

# Load the tokenizer and text encoder to tokenize and encode the text.
clip_model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(torch_device);
vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name).to(torch_device);
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet").to(torch_device);

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                 beta_end=0.012,
                                 beta_schedule="scaled_linear",
                                 num_train_timesteps=1000)


def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    return error

prompt = 'A campfire (oil on canvas)' #@param
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 50  #@param           # Number of denoising steps
guidance_scale = 8 #@param               # Scale for classifier-free guidance
generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
batch_size = 1
blue_loss_scale = 200 #@param

text_input = tokenizer([prompt],
                       padding="max_length",
                       max_length=tokenizer.model_max_length,
                       truncation=True,
                       return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# And the uncond. input as before:
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Prep Scheduler
set_timesteps(scheduler, num_inference_steps)

# Prep latents
latents = torch.randn(
  (batch_size, unet.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to(torch_device)
latents = latents * scheduler.init_noise_sigma


#
    # text_input = tokenizer([textual_direction],
    #                        padding="max_length",
    #                        max_length=tokenizer.model_max_length,
    #                        truncation=True,
    #                        return_tensors="pt")
# cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    # return cos_similarity(img_embed, text_embed)[0]
def cosine_loss(gen_image):
    textual_direction = "evening sunlight"
    inputs = processor(text=textual_direction,
                       return_tensors="pt",
                       padding=True)
    with torch.no_grad():
        text_embed = CLIPTextModelWithProjection.from_pretrained(
            clip_model_name)(**inputs).text_embeds
    gen_image_clamped = gen_image.clamp(0, 1).mul(255).round()
    inputs = processor(images=gen_image_clamped,
                       return_tensors="pt",
                       padding=True)
    image_embed = CLIPVisionModelWithProjection.from_pretrained(
        clip_model_name)(**inputs).image_embeds
    image_embed.requires_grad = True
    similarity = nn.functional.cosine_similarity(text_embed, image_embed)[0]
    print(similarity)
    return 1 - similarity

# def cosine_loss(gen_image):
#     gen_image_clamped = gen_image.clamp(0, 1).mul(255).round()
#     textual_direction = "evening lightning"
#     inputs = processor(text=textual_direction,
#                        images=gen_image_clamped,
#                        return_tensors="pt",
#                        padding=True)
#     return 1 - clip_model(**inputs).logits_per_image
#

# Loop
for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    sigma = scheduler.sigmas[i]
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

    # perform CFG
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #### ADDITIONAL GUIDANCE ###
    if i%5 == 0:
        # Requires grad on the latents
        latents = latents.detach().requires_grad_()

        # Get the predicted x0:
        latents_x0 = latents - sigma * noise_pred
        # stepper = scheduler.step(noise_pred, t, latents)
        # latents_x0 = stepper.pred_original_sample

        # Decode to image space
        denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)

        # Calculate loss
        # loss = blue_loss(denoised_images) * blue_loss_scale
        loss = cosine_loss(denoised_images) * blue_loss_scale

        # Occasionally print it out
        if (i + 1) % 10==0:
            print(i, 'loss:', loss.item())

        # Get gradient
        cond_grad = torch.autograd.grad(loss, latents)[0]
        # , allow_unused=True

        # Modify the latents based on this gradient
        latents = latents.detach() - cond_grad * sigma**2
        # scheduler._step_index = scheduler._step_index - 1

        # Now step with scheduler
    latents = scheduler.step(noise_pred, t, latents).prev_sample

latents_to_pil(latents)[0].show()
