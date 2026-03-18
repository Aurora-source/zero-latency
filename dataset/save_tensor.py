import os
import json
import torch

data_path = "dataset/v1.0-mini/v1.0-mini"
save_path = "dataset/processed"

os.makedirs(save_path, exist_ok=True)

files = [f for f in os.listdir(data_path) if "ego_pose" in f]

count = 0

for file in files:
    file_path = os.path.join(data_path, file)

    with open(file_path, 'r') as f:
        data = json.load(f)

    coords = []

    for item in data[:10]:
        x = item["translation"][0]
        y = item["translation"][1]
        coords.append([x, y])

    tensor = torch.tensor(coords)

    torch.save(tensor, os.path.join(save_path, f"sample_{count}.pt"))

    count += 1

print("Saved", count, "tensor files")