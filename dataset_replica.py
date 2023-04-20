import numpy as np
import os
from PIL import Image
import json
import torch
from torchvision import transforms
from tqdm import tqdm, trange

#with Image.open("hopper.jpg") as im:

def read_scene(path, img_size=(128,128)):
    transform_rgb = [transforms.ToTensor()]
    if img_size is not None:
        transform_rgb.insert(0,transforms.Resize(img_size))
    transform_rgb = transforms.Compose(transform_rgb)
    pose_path = os.path.join(path, "cameras.json")
    rgb_data = []
    pose_data = []
    with open(pose_path) as file:
        cam_content = json.loads(file.read())
        obs_size = len(cam_content)
        for i in range(obs_size):
            rgb_fname = str(i).zfill(3) + "_rgb.png"
            rgb_pil = Image.open(os.path.join(path, rgb_fname)).convert('RGB')
            rgb_tensor = transform_rgb(rgb_pil)
            rgb_data.append(rgb_tensor.unsqueeze(0))
            pose_tensor = torch.tensor(cam_content[i]['Rt']).reshape((-1))[:12]
            pose_data.append(pose_tensor.unsqueeze(0))

    rgb_tensor_scene = torch.cat(rgb_data, dim=0)
    pose_tensor_scene = torch.cat(pose_data, dim=0)
    #print(rgb_tensor_scene.shape)
    #print(pose_tensor_scene.shape)
    return rgb_tensor_scene, pose_tensor_scene

def load_replica(path, img_size=(128,128), device='cpu'):
    scene_len = len(os.listdir(path))
    rgb_dataset = []
    pose_dataset = []
    for i in tqdm(range(scene_len)):
        scene_path = os.path.join(path, str(i).zfill(2))
        rgb, pose = read_scene(scene_path, img_size)
        rgb_dataset.append(rgb.unsqueeze(0))
        pose_dataset.append(pose.unsqueeze(0))
    rgb_tensor_dataset = torch.cat(rgb_dataset, dim=0)
    pose_tensor_dataset = torch.cat(pose_dataset, dim=0)
    return rgb_tensor_dataset.to(device), pose_tensor_dataset.to(device)

if __name__ == "__main__":
    read_scene("E:/ml-gsn/data/replica_all/train/00")
    load_replica("E:/ml-gsn/data/replica_all/train")
    
    #color_data, pose_data = read_dataset_color("Datasets/MazeBoardRandom/")
    #img_obs, pose_obs, img_query, pose_query = get_batch(color_data, pose_data, to_torch=False)
    #print(img_obs.shape, pose_obs.shape, img_query.shape, pose_query.shape)
