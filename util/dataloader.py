import os
import random

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from transformers import CLIPImageProcessor
from torch.utils.data import DataLoader


class FaceDataset(Dataset):
    def __init__(self, config: dict):
        self.config = config
        self.path = config['path']
        self.clip_image_processor: CLIPImageProcessor = config['clip_image_processor']
        self.file_list = [os.path.join(self.path, file) for file in os.listdir(self.path) if
                          file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        self.face_dict = {}

        for file in self.file_list:
            person_name = self.get_person_name_from_file_path(file)
            if person_name not in self.face_dict:
                self.face_dict[person_name] = []
            self.face_dict[person_name].append(file)

        # this might take a while
        print(f"  -  Preprocessing image dimensions")
        new_file_list = []
        bad_count = 0
        for file in tqdm(self.file_list):
            new_file_list.append(file)

        self.file_list = new_file_list

        print(f"  -  Found {len(self.file_list)} images")
        assert len(self.file_list) > 0, f"no images found in {self.path}"

    def __len__(self):
        return len(self.file_list)

    def get_person_name_from_file_path(self, file_path):
        filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        return filename_no_ext.split('_')[0]

    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        try:
            img = exif_transpose(img)
        except Exception as e:
            print(f"Error opening image: {img_path}")
            print(e)
            # make a noise image if we can't open it
            img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
        img = self.clip_image_processor(img, return_tensors="pt").data['pixel_values'][0]
        return img

    def __getitem__(self, index):
        person_name = self.get_person_name_from_file_path(self.file_list[index])
        anchor = self.get_image(self.file_list[index])
        positive = self.get_image(random.choice(self.face_dict[person_name]))
        negative = self.get_image(random.choice(self.file_list))

        # stack on axis 1
        img = torch.cat([anchor, positive, negative], dim=1)

        return img


def get_dataloader(config: dict):
    face_dataset = FaceDataset(config)
    return DataLoader(face_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)