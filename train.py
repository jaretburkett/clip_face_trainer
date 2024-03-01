import os

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate import Accelerator
from util.dataloader import get_dataloader
from tqdm import tqdm

accelerator = Accelerator()

clip_path = "openai/clip-vit-large-patch14-336"
save_path_root = os.path.join("output")
os.makedirs(save_path_root, exist_ok=True)
save_path = os.path.join(save_path_root, 'model')
device = accelerator.device

if os.path.exists(save_path):
    print(f"Loading model from {save_path}")
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(save_path, torch_dtype=torch.bfloat16).to(device)
else:
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_path, torch_dtype=torch.bfloat16).to(device)

# Assuming model is already defined and available
image_processor = CLIPImageProcessor.from_pretrained(clip_path)
dataloader = get_dataloader({
    "path": "/mnt/Storage/Datasets/aligned_face/",
    "clip_image_processor": image_processor,
    "batch_size": 1
})

# Define triplet loss
triplet_loss = nn.TripletMarginLoss(margin=5, p=2, eps=1e-7)

# Optimizer
optimizer = optim.AdamW(vision_encoder.parameters(), lr=1e-6)

image_processor, optimizer, dataloader = accelerator.prepare(
    image_processor, optimizer, dataloader
)


def save_model():
    global vision_encoder
    print(f"Saving model to {save_path}")
    vision_encoder.save_pretrained(save_path)


save_every_n_steps = 1000
n_step = 0
pbar = tqdm(range(len(dataloader)), desc="Training")
# Training loop
num_epochs = 5
running_avg_loss = []
for epoch in range(num_epochs):
    vision_encoder.train()
    running_loss = 0.0
    for data in dataloader:
        # In a real scenario, data would come from your DataLoader
        anchor, positive, negative = data.chunk(3, dim=2)

        batch = torch.cat([anchor, positive, negative], dim=0)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass to get output embeddings
        output = vision_encoder(batch)['image_embeds']
        anchor_out, positive_out, negative_out = output.chunk(3, dim=0)

        # Calculate loss
        loss = triplet_loss(anchor_out.float(), positive_out.float(), negative_out.float())
        # Backward pass and optimize
        accelerator.backward(loss)
        optimizer.step()

        # set the description to the loss
        # format as exponent
        running_avg_loss.append(loss.item())
        if len(running_avg_loss) > 20:
            running_avg_loss.pop(0)

        formatted_loss = "{:.2e}".format(sum(running_avg_loss) / len(running_avg_loss))

        # formatted_loss = "{:.2e}".format(loss.item())
        pbar.set_description(f"Loss: {formatted_loss}")

        running_loss += loss.item()
        n_step += 1
        if n_step % save_every_n_steps == 0:
            save_model()

        pbar.update(1)

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

