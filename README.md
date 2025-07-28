# Iranian Vehicle Classifier 🚗🇮🇷



**This project is developed to classify **20 types of Iranian vehicles**, including both **trucks** and **passenger cars**, using a **custom CNN model built from scratch in PyTorch**.**



## 🚀 Project Summary



**The goal of this project is to accurately identify close-up images of Iranian vehicles from traffic camera views. Unlike traditional datasets that include full street scenes, our model assumes that the **input images are cropped vehicles**, extracted from object detection bounding boxes — ensuring focused, high-quality classification.**

---
# ⚠️ Notes
**This model is specifically optimized for cropped vehicle images taken from front view traffic cameras.
The model does not perform well on images containing multiple objects, street scenes, or non-vehicle content.
Intended use: vehicle classification after detection step.**
# 🌐 Gradio Demo (Live Test)

 **🚦 Try the model live using the link below:**
#        ---------------------------------->     [Test Live](https://huggingface.co/spaces/Alirezachahardoli/Tannan1) <------------------------------



## 📊 Dataset



- ✅ **Number of Classes:** 21 (Saina,Pride, Pars, Dena, Samand,L90, Renualt-Trucks, Vans, etc.)

- 📈 **Balanced Dataset:** Yes

- 🖼️ **Image Type:** Close-up vehicle front-view, realistic traffic camera angles

- 🔗 **Dataset Link:** **[Dataset](https://www.kaggle.com/datasets/alirezachahardoli/iranian-car-imageclassification)**



---



## 🧠 Model Architecture



A **custom Convolutional Neural Network (CNN)** was designed and trained **from scratch** in PyTorch with the following features:



- 🧩 **Skip Connections** (inspired by ResNet)

- 🔁 **ReduceLROnPlateau** scheduler for automatic LR adjustment

- 🔥 High performance:  

  - **Train Accuracy:** ~99%  

  - **Test Accuracy:** ~99%  

  - **Loss:** Extremely low

- ⚡️ Inference speed: ~**5.05 ms per 100 images**



---



## 🧪 Training Configuration



- 🗂️ Trained using custom PyTorch dataloaders

- 🛠️ Augmentation applied using `torchvision.transforms`

- 📉 Learning rate automatically adjusted when validation loss plateaus

- 🔍 Evaluation includes confusion matrix, accuracy, and loss visualization



---



## 🖥️ Files Structure


```bash
├── app.py                  # Gradio deployment

├── augmentation.py         # Custom augmentations

├── config.py               # Hyperparameters and paths

├── custom_model.py         # CNN model with skip connections

├── dataset.py              # create train_dataloader & test_dataloader with `from torch.utils.data import DataLoader`

├── main.py                 # Training pipeline

├── Train_Test_loop.py      # Train & test functions

├── model with info.pt      # Trained model

├── notebook.ipynb          # Kaggle training & visualization

├── requirements.txt        # Dependencies

└── README.md
```

---


            


- 🔗 Gradio Deployment Link

- 🧪 Upload a close-up image of a vehicle for instant classification!

- 🛠 How to Run


1. Clone the Repository


## installa Requirements

pip install -r requirements.txt
