import os
import glob
import json

from os.path import join
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

class DatasetMSDAll(Dataset):
    def __init__(self, datapath, fold, transform, split, shot):
        super().__init__()
        self.split = 'test' if split in ['val', 'test'] else 'trn'
        self.benchmark = 'msd'
        self.shot = shot
        self.num = 0

        self.base_path = datapath
        # Use Tr or Ts based on split
        self.suffix = 'Tr' if self.split == 'trn' else 'Ts'
        self.transform = transform

        self.categories = []
        for task in os.listdir(datapath):
            if task == ".":
                continue
            
            with open(join(datapath, task, "dataset.json"), "r") as f:
                dataset = json.load(f)
            self.categories.extend([f"{task}_{cat}" for cat in dataset["labels"].keys() if cat != "0"])
        self.class_ids = range(0, len(self.categories))
        self.nclass = len(self.categories)
        
        # Build metadata for sampling
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        
        # Verify we have enough images for the requested shots
        min_images = min(len(imgs) for imgs in self.img_metadata_classwise.values())
        if min_images < self.shot + 1:  # Need at least shot+1 for query and support
            raise ValueError(f"Not enough images! Found {min_images} images but need at least {self.shot + 1} for {self.shot}-shot learning")
    
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)
        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'class_id': self.categories[class_sample]
        }
        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]
        
        # Extract sample folder and slice name (e.g., from imagesTr/STS_001/slice_005.png)
        # Get the corresponding mask path
        query_parts = query_name.split('/')
        task = query_parts[-4]
        sample_folder = query_parts[-2]  # e.g., STS_001
        slice_name = query_parts[-1]      # e.g., slice_005.png
        query_mask_path = join(self.base_path, task, f"labels{self.suffix}", sample_folder, slice_name)
        
        support_mask_paths = []
        for support_name in support_names:
            support_parts = support_name.split('/')
            task = support_parts[-4]
            sample_folder = support_parts[-2]
            slice_name = support_parts[-1]
            support_mask_path = join(self.base_path, task, f"labels{self.suffix}", sample_folder, slice_name)
            support_mask_paths.append(support_mask_path)
        
        query_mask = self.read_mask(query_mask_path)
        support_masks = [self.read_mask(path) for path in support_mask_paths]
        
        return query_img, query_mask, support_imgs, support_masks
    

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask_out = torch.zeros_like(mask)
        mask_out[mask == 0] = 0      # background
        mask_out[mask > 0] = 1
        return mask_out
    
    def sample_episode(self, idx):
        class_id = 0
        class_sample = self.categories[0]
        cum_sum = 0
        for i, (cat, data) in enumerate(self.img_metadata_classwise.items()):
            class_id = i
            class_sample = cat
            
            if idx < cum_sum + len(data):
                break
            
            cum_sum += len(data)
            
        query_name = self.img_metadata_classwise[class_sample][idx - cum_sum]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break
        
        return query_name, support_names, class_id
    
    def build_img_metadata_classwise(self):
        """Build metadata: collect all image paths with valid masks."""
        # For sarcoma, we'll use a simpler approach: all slices with any annotation
        # are valid for all categories (tumor and necrosis are both valid targets)
        img_metadata_classwise = {cat:[] for cat in self.categories}
        
        for task in os.listdir(self.base_path):
            if task == ".":
                continue
            
            # Scan all sample folders
            task_path = join(self.base_path, task)
            sample_folders = sorted(glob.glob(join(task_path, f"images{self.suffix}", '*')))
            for sample_folder in tqdm(sample_folders, desc=task, leave=False):
                if not os.path.isdir(sample_folder):
                    continue
                
                sample_name = os.path.basename(sample_folder)
                label_folder = join(task_path, f"labels{self.suffix}", sample_name)
                
                # print(label_folder)
                if not os.path.exists(label_folder):
                    continue
                
                # Get all PNG files in this sample folder
                img_files = sorted(glob.glob(join(sample_folder, '*.png')))
                
                for img_path in img_files:
                    slice_name = os.path.basename(img_path)
                    label_path = join(label_folder, slice_name)
                    
                    class_id = int(float(os.path.splitext(label_path)[0].split("_")[-1]))
                    if class_id == 0:
                        continue
                    
                    if not os.path.exists(label_path):
                        continue
                    
                    # Read mask to see if it has any annotation
                    mask = np.array(Image.open(label_path).convert('L'))
                    
                    # Check if mask has any non-zero values (any annotation)
                    if np.any(mask > 0):
                        img_metadata_classwise[f"{task}_{class_id}"].append(img_path)
                        # print(img_path, label_path)
                        self.num += 1
        
        # Assign all valid images to each category
        # This allows sampling from any category to work
         
        print(f"Found {self.num} valid slices with annotations")    

        return img_metadata_classwise

def build(image_set, args):
    # Use configurable image size if provided, otherwise default to 518
    img_size = getattr(args, 'img_size', 256)
    
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DatasetMSDAll(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset