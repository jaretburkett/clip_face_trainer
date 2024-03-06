import os

import torch
import torch.nn as nn
from bitsandbytes import optim
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipForImageClassification, SiglipImageProcessor
from accelerate import Accelerator
from util.dataloader_insight import get_dataloader
from tqdm import tqdm
from prodigyopt import Prodigy
import wandb

run = wandb.init(project="clip_insight_face_trainer")
config = run.config

accelerator = Accelerator()

# clip_path = "openai/clip-vit-large-patch14-336"
# clip_path = "ostris/photo-maker-face-sdxl"
clip_path = "/mnt/Models/stable-diffusion/models/control-net-models/IP-Adapter/CLIP-H-Face-v2/"
# clip_path = "google/siglip-base-patch16-512"
save_path_root = os.path.join("output")
os.makedirs(save_path_root, exist_ok=True)
save_path = os.path.join(save_path_root, 'insight_face_clip_h_model')
device = accelerator.device

if 'siglip' in clip_path.lower():
    ProcessorClass = SiglipImageProcessor
    ModelClass = SiglipForImageClassification
else:
    ProcessorClass = CLIPImageProcessor
    ModelClass = CLIPVisionModelWithProjection


if os.path.exists(save_path):
    print(f"Loading model from {save_path}")
    vision_encoder = ModelClass.from_pretrained(
        save_path,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16
    ).to(device)
else:
    vision_encoder = ModelClass.from_pretrained(
        clip_path,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16
    ).to(device)

vision_encoder: CLIPVisionModelWithProjection = vision_encoder
# vision_encoder.gradient_checkpointing_enable()
vision_encoder.train()

print(f"Compiling model")
vision_encoder = torch.compile(vision_encoder, mode='default')

# Assuming model is already defined and available
image_processor = ProcessorClass.from_pretrained(clip_path)
dataloader = get_dataloader({
    # "path": "/mnt/Datasets/face_pairs2/control_clean/",
    "path": "/mnt/Datasets/FlickerFaceHQ_images1024x1024/images1024x1024/",
    "clip_image_processor": image_processor,
    "batch_size": 20
})

# Optimizer
optimizer = optim.AdamW8bit(vision_encoder.parameters(), lr=5e-5, weight_decay=1e-3)
# optimizer = Prodigy(
#     vision_encoder.parameters(),
#     lr=1.,
#     weight_decay=1e-3,
#     decouple=True,
#     d0=1e-5
# )

image_processor, optimizer, dataloader = accelerator.prepare(
    image_processor, optimizer, dataloader
)


def save_model():
    global vision_encoder
    print(f"Saving model to {save_path}")
    vision_encoder.save_pretrained(save_path)
    image_processor.save_pretrained(save_path)


log_every = 1

save_every_n_steps = 1000
n_step = 0
num_epochs = 500
pbar = tqdm(range(len(dataloader) * num_epochs), desc="Training")
running_loss = 0.0
running_loss_steps = 0
mse_loss = nn.MSELoss()
# Training loop
running_avg_loss = []
for epoch in range(num_epochs):
    vision_encoder.train()
    running_loss = 0.0
    for data in dataloader:
        # In a real scenario, data would come from your DataLoader
        with torch.no_grad():
            clip_img, faceid_embeds = data
            clip_img = clip_img.to(torch.bfloat16).detach()
            faceid_embeds = faceid_embeds.to(torch.bfloat16).detach()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # image embeds loss
        clip_image_embeds = vision_encoder(clip_img)['image_embeds']

        if clip_image_embeds.shape[1] != faceid_embeds.shape[1]:
            if clip_image_embeds.shape[1] // 2 == faceid_embeds.shape[1]:
                # get every other one
                clip_image_embeds1 = clip_image_embeds[:, ::2]
                # the other half
                clip_image_embeds2 = clip_image_embeds[:, 1::2]

                clip_image_embeds = (clip_image_embeds1 + clip_image_embeds2) / 2

        # mse on just positive
        loss = mse_loss(clip_image_embeds.float(), faceid_embeds.float())

        # Backward pass and optimize
        accelerator.backward(loss)
        optimizer.step()

        with torch.no_grad():
            # set the description to the loss
            # format as exponent
            running_avg_loss.append(loss.item())
            if len(running_avg_loss) > 20:
                running_avg_loss.pop(0)

            # learning_rate = (
            #         optimizer.param_groups[0]["d"] *
            #         optimizer.param_groups[0]["lr"]
            # )
            learning_rate = (
                    # optimizer.param_groups[0]["d"] *
                    optimizer.param_groups[0]["lr"]
            )

            formatted_learning_rate = "{:.2e}".format(learning_rate)

            avg_loss = sum(running_avg_loss) / len(running_avg_loss)

            formatted_loss = "{:.2e}".format(avg_loss)

            # formatted_loss = "{:.2e}".format(loss.item())
            pbar.set_description(f"loss: {formatted_loss}, lr: {formatted_learning_rate}", refresh=True)

            running_loss += loss.item()
            n_step += 1
            running_loss_steps += 1
            if n_step % save_every_n_steps == 0:
                save_model()

            if n_step % log_every == 0:
                wandb.log({"loss": avg_loss, "learning_rate": learning_rate})

        pbar.update(1)

    epoch_loss = running_loss/running_loss_steps
    wandb.log({"epoch_loss": epoch_loss})
    print(f"Epoch {epoch+1}, Loss: {running_loss/running_loss_steps}")
    running_loss = 0.0
    running_loss_steps = 0

