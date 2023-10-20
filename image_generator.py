import os
from pathlib import Path
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging

from utils import load_embedding_bin, set_timesteps, latents_to_pil
from loss import blue_loss, cosine_loss

torch.manual_seed(11)
logging.set_verbosity_error()

# Set device
torch_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
if "mps" == torch_device:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Style embeddings
STYLE_EMBEDDINGS = {
    "illustration-style": "illustration_style.bin",
    "line-art": "line-art.bin",
    "hitokomoru-style": "hitokomoru-style.bin",
    "midjourney-style": "midjourney-style.bin",
    "hanfu-anime-style": "hanfu-anime-style.bin",
    "birb-style": "birb-style.bin",
    "style-of-marc-allante": "Marc Allante.bin",
}
LOSS = {"blue_loss": blue_loss,
        "cosine_loss": cosine_loss}
STYLE_SEEDS = [11, 56, 110, 65, 5, 29, 47]
# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae"
).to(torch_device)
#
# # Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14").to(torch_device)

# # The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
).to(torch_device)
#
# # The noise scheduler
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)

token_emb_layer = text_encoder.text_model.embeddings.token_embedding
pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
position_embeddings = pos_emb_layer(position_ids)


def build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


def get_output_embeds(input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = build_causal_attention_mask(
        bsz, seq_len, dtype=input_embeddings.dtype
    )

    # Getting the output embeddings involves calling the model with passing output_hidden_states=True
    # so that it doesn't just return the pooled final predictions:
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None,  # We aren't using an attention mask so that can be None
        causal_attention_mask=causal_attention_mask.to(torch_device),
        output_attentions=None,
        output_hidden_states=True,  # We want the output embs not the final output
        return_dict=None,
    )

    # We're interested in the output hidden state only
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = text_encoder.text_model.final_layer_norm(output)

    # And now they're ready!
    return output


# Generating an image with these modified embeddings
def generate_with_embs(text_embeddings, seed, max_length):
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 30  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)
    batch_size = 1

    # tokenizer
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(scheduler, num_inference_steps)

    # Prep latents
    # step = " prep_latents "
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), 
                     total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents_to_pil(latents)[0]


def generate_image_from_embeddings(
        mod_output_embeddings, seed, max_length, 
        loss_selection, additional_prompt):
    height = 512
    width = 512
    num_inference_steps = 50
    guidance_scale = 8
    generator = torch.manual_seed(seed)
    batch_size = 1
    if loss_selection == "blue_loss":
        loss_fn = LOSS["blue_loss"]
        loss_scale = 120
    else:
        loss_fn = LOSS["cosine_loss"](additional_prompt)
        loss_scale = 20

    # Use the modified_output_embeddings directly
    text_embeddings = mod_output_embeddings

    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(scheduler, num_inference_steps)

    # Prep latents
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), 
                     total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

        # perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        #### ADDITIONAL GUIDANCE ###
        if i % 5 == 0:
            # Requires grad on the latents
            latents = latents.detach().requires_grad_()

            # Get the predicted x0:
            # latents_x0 = latents - sigma * noise_pred
            latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample
            scheduler._step_index -= 1
            # Decode to image space
            denoised_images = (
                vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5
            )  # range (0, 1)

            # Calculate loss
            loss = loss_fn(denoised_images) * loss_scale

            # Occasionally print it out
            if i % 10 == 0:
                print(i, "loss:", loss.item())

            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents)[0]

            # Modify the latents based on this gradient
            latents = latents.detach() - cond_grad * sigma**2

        # Now step with scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents_to_pil(latents)[0]


def generate_image_per_style(prompt, style_embed, style_seed, style_embedding_key):
    modified_output_embeddings = None
    gen_out_style_image = None
    max_length = 0

    # Tokenize
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_input.input_ids.to(torch_device)

    # Get token embeddings
    token_embeddings = token_emb_layer(input_ids)

    replacement_token_embedding = style_embed[style_embedding_key]

    # Insert this into the token embeddings
    token_embeddings[
        0, torch.where(input_ids[0] == 6829)[0]
    ] = replacement_token_embedding.to(torch_device)

    # Combine with pos embs
    input_embeddings = token_embeddings + position_embeddings

    #  Feed through to get final output embs
    modified_output_embeddings = get_output_embeds(input_embeddings)

    # And generate an image with this:
    max_length = text_input.input_ids.shape[-1]

    gen_out_style_image = generate_with_embs(
        modified_output_embeddings, style_seed, max_length
    )

    return gen_out_style_image


def generate_image_per_loss(
        prompt, style_embed, style_seed, style_embedding_key,
        loss, additional_prompt
    ):
    gen_out_loss_image = None

    # Tokenize
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_input.input_ids.to(torch_device)

    # Get token embeddings
    token_embeddings = token_emb_layer(input_ids)

    replacement_token_embedding = style_embed[style_embedding_key].to(torch_device)

    # Insert this into the token embeddings
    token_embeddings[
        0, torch.where(input_ids[0] == 6829)[0]
    ] = replacement_token_embedding

    # Combine with pos embs
    input_embeddings = token_embeddings + position_embeddings
    modified_output_embeddings = get_output_embeds(input_embeddings)

    # max_length = tokenizer.model_max_length

    max_length = text_input.input_ids.shape[-1]
    gen_out_loss_image = generate_image_from_embeddings(
        modified_output_embeddings, style_seed, max_length,
        loss, additional_prompt
    )

    return gen_out_loss_image


def generate_image_per_prompt_style(text_in, style_in, 
                                    loss, additional_prompt):
    gen_style_image = None
    gen_loss_image = None
    STYLE_KEYS = []
    style_key = ""

    if style_in not in STYLE_EMBEDDINGS:
        raise ValueError(
            f"Unknown style: {style_in}. Available styles are: {', '.join(STYLE_EMBEDDINGS.keys())}"
        )

    STYLE_SEEDS = [32, 64, 128, 16, 8, 96]
    STYLE_KEYS = list(STYLE_EMBEDDINGS.keys())
    print(f"prompt: {text_in}")
    print(f"style: {style_in}")

    idx = STYLE_KEYS.index(style_in)
    style_file = STYLE_EMBEDDINGS[style_in]
    print(f"style_file: {style_file}")

    prompt = text_in

    style_seed = STYLE_SEEDS[idx]

    style_key = Path(style_file).stem
    style_key = style_key.replace("_", "-")
    print(style_key, STYLE_KEYS, style_file)

    file_path = os.path.join(os.getcwd(), style_file)
    embedding = load_embedding_bin(file_path)
    style_key = f"<{style_key}>"

    gen_style_image = generate_image_per_style(prompt, embedding, style_seed, style_key)

    gen_loss_image = generate_image_per_loss(prompt, embedding, style_seed, style_key, loss, additional_prompt)

    return [gen_style_image, gen_loss_image]
