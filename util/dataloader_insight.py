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
from insightface.app import FaceAnalysis
import cv2
from safetensors.torch import load_file, save_file

# app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], rcond=None)


def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


class FaceInsightDataset(Dataset):
    def __init__(self, config: dict):
        self.config = config
        self.path = config['path']
        self.clip_image_processor: CLIPImageProcessor = config['clip_image_processor']
        self.file_list = [os.path.join(self.path, file) for file in os.listdir(self.path) if
                          file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        self.app = None
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


    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        try:
            img = exif_transpose(img)
        except Exception as e:
            print(f"Error opening image: {img_path}")
            print(e)
            # make a noise image if we can't open it
            img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
        return img

    def __getitem__(self, index):
        try:
            img = self.get_image(self.file_list[index])
            clip_img = self.clip_image_processor(img, return_tensors="pt").data['pixel_values'][0]

            filename_no_ext = os.path.splitext(os.path.basename(self.file_list[index]))[0]
            face_id_path = os.path.join(os.path.dirname(self.file_list[index]), filename_no_ext + ".fid")

            faceid_embeds = None
            if os.path.exists(face_id_path):
                faceid_embeds = load_file(face_id_path)['embeds']

            if faceid_embeds is None:
                # add mirroerd padding of 200px on all sides to img
                img = pil_to_cv2(img)
                # padding is 40% of the image size
                padding = int(img.shape[0] * 0.4)
                img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

                if self.app is None:
                    self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'], rcond=None)
                    self.app.prepare(ctx_id=0, det_size=(640, 640))
                faces = self.app.get(img, max_num=1)

                faceid_embeds = torch.from_numpy(faces[0].normed_embedding)
                save_file({'embeds': faceid_embeds}, face_id_path)

            return clip_img, faceid_embeds
        except Exception as e:
            print(f"Error getting image: {e}")
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))


def get_dataloader(config: dict):
    face_dataset = FaceInsightDataset(config)
    return DataLoader(face_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
