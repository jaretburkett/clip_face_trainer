import os

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipForImageClassification, SiglipImageProcessor
from accelerate import Accelerator
from util.dataloader import get_dataloader
from tqdm import tqdm
from prodigyopt import Prodigy
import wandb

run = wandb.init(project="clip_face_trainer")
config = run.config

accelerator = Accelerator()

# clip_path = "openai/clip-vit-large-patch14-336"
clip_path = "ostris/photo-maker-face-sdxl"
# clip_path = "google/siglip-base-patch16-512"
save_path_root = os.path.join("output")
os.makedirs(save_path_root, exist_ok=True)
save_path = os.path.join(save_path_root, 'model')
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
    "path": "/mnt/Datasets/face_pairs2/control_clean/",
    "clip_image_processor": image_processor,
    "batch_size": 8
})


# Define triplet loss
triplet_loss = nn.TripletMarginLoss(margin=1, p=2, eps=1e-7)

# Optimizer
# optimizer = optim.AdamW(vision_encoder.parameters(), lr=1e-6)
optimizer = Prodigy(
    vision_encoder.parameters(),
    lr=1.,
    weight_decay=1e-3,
    decouple=True,
    d0=1e-7
)

image_processor, optimizer, dataloader = accelerator.prepare(
    image_processor, optimizer, dataloader
)


def save_model():
    global vision_encoder
    print(f"Saving model to {save_path}")
    vision_encoder.save_pretrained(save_path)
    image_processor.save_pretrained(save_path)


log_every = 10

save_every_n_steps = 1000
n_step = 0
num_epochs = 100
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
            anchor, positive, negative = data.chunk(3, dim=2)
            batch = torch.cat([anchor, positive, negative], dim=0).to(torch.bfloat16).detach()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass to get output embeddings
        # output = vision_encoder(batch)['image_embeds']
        output_last_hidden_state = vision_encoder(batch)[-1]
        anchor_output_last_hidden_state, positive_output_last_hidden_state, negative_output_last_hidden_state = output_last_hidden_state.chunk(3, dim=0)

        # Calculate loss
        loss_last_hidden_state = triplet_loss(
            anchor_output_last_hidden_state.float(),
            positive_output_last_hidden_state.float(),
            negative_output_last_hidden_state.float()
        )


        # image embeds loss
        output_image_embeds = vision_encoder(batch)['image_embeds']
        anchor_output_image_embeds, positive_output_image_embeds, negative_output_image_embeds = output_image_embeds.chunk(3, dim=0)

        loss_image_embeds = triplet_loss(
            anchor_output_image_embeds.float(),
            positive_output_image_embeds.float(),
            negative_output_image_embeds.float()
        )

        # mse on just positive
        mse_loss_positive = mse_loss(anchor_output_image_embeds.float(), positive_output_image_embeds.float())

        # apply negative pressure
        mse_loss_negative = mse_loss(anchor_output_image_embeds.float(), negative_output_image_embeds.float()) * 0.1

        total_mse_loss = mse_loss_positive - mse_loss_negative

        loss = loss_last_hidden_state + loss_image_embeds + total_mse_loss

        # Backward pass and optimize
        accelerator.backward(loss)
        optimizer.step()

        with torch.no_grad():
            # set the description to the loss
            # format as exponent
            running_avg_loss.append(loss.item())
            if len(running_avg_loss) > 20:
                running_avg_loss.pop(0)

            learning_rate = (
                    optimizer.param_groups[0]["d"] *
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

