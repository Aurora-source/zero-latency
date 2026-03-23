# NuScenes Dataset Loader (PyTorch)

This module loads the nuScenes mini dataset and converts ego trajectory data into PyTorch tensors.

---

## 📁 Folder Structure
zero-latency/
├── dataset/
│ ├── v1.0-mini/ # raw dataset
│ ├── processed/ # saved tensor files (.pt)
│ ├── pytorch_dataset.py
│ ├── save_tensor.py
├── test_loader.py


---

## 🚀 Step 1: Dataset

Place nuScenes mini dataset inside: dataset/v1.0-mini/


---

## ⚙️ Step 2: Generate Tensor Files

Run: python dataset/save_tensor.py


This will:
- Read `ego_pose.json`
- Extract trajectory (x, y)
- Save tensor files in: dataset/processed/


---

## 🧠 Step 3: Load Dataset

Run: python test_loader.py


Output example: {'trajectory': tensor([...])}


---

## 📌 Notes

- Only ego trajectory is processed for now
- Each tensor contains (x, y) coordinates
- Ready for model training

---




