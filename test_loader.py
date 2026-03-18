from torch.utils.data import DataLoader
from dataset.pytorch_dataset import NuScenesDataset

dataset = NuScenesDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in loader:
    print(batch)
    break