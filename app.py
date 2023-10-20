import gradio as gr

import os
import torch
from image_generator import generate_image_per_prompt_style

torch.manual_seed(11)


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
# Define Interface
title = "Generative Art - Stable Diffusion with Styles and Blue Loss"

gr_interface = gr.Interface(
    generate_image_per_prompt_style,
    inputs=[
        gr.Textbox("A flying bird", label="Prompt"),
        gr.Dropdown(
            [
                "illustration_style",
                "line-art",
                "hitokomoru-style",
                "midjourney-style",
                "hanfu-anime-style",
                "birb-style",
                "style-of-marc-allante",
            ],
            value="illustration-style",
            label="Pre-trained Styles",
        ),
        gr.Dropdown(
            [
                "blue_loss",
                "additional_context",
            ],
            value="additional_context",
            label="Additional guidance for image generation",
        ),
    ],
    outputs=[
        gr.Gallery(
            label="Generated images",
            show_label=False,
            elem_id="gallery",
            columns=[2],
            rows=[2],
            object_fit="contain",
            height="auto",
        )
    ],
    title=title,
)
gr_interface.launch(debug=True)
