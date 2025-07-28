# Iranian Vehicle Classifier 🚗🇮🇷



**This project is developed to classify **20 types of Iranian vehicles**, including both **trucks** and **passenger cars**, using a **custom CNN model built from scratch in PyTorch**.**



## 🚀 Project Summary



**The goal of this project is to accurately identify close-up images of Iranian vehicles from traffic camera views. Unlike traditional datasets that include full street scenes, our model assumes that the **input images are cropped vehicles**, extracted from object detection bounding boxes — ensuring focused, high-quality classification.**

---
# ⚠️ Notes:
**This model is specifically optimized for cropped vehicle images taken from front view traffic cameras.
The model does not perform well on images containing multiple objects, street scenes, or non-vehicle content.
Intended use: vehicle classification after detection step.**
**The model is trained on** *224x224* **RGB images. input images must be Normalized using the following mean and std values.(as used during training):**
```python
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        transforms.Normalize(mean=(0.5,0.5,0.5),
                      std=(0.5,0.5,0.5))
    ])
```

# 🌐 Gradio Demo Deployment(Live Test)
**The model is deployed using Gradio, allowing users to test it online by uploading an image of a close-up Iranian vehicle.**

 **🚦 Try the model live using the link below:**
#        --------Click-------> [Live Test](https://huggingface.co/spaces/Alirezachahardoli/Tannan1) 



## 📊 Dataset
# ⚠️ Notes:
  **Before training, be sure to run**  *`augment_and_save()` **from** `augmentation.py`* **to generate the augmented dataset**



- ✅ **Number of Classes:** 20 (Saina,Pride, Pars, Dena, Samand,L90, Renualt-Trucks, Vans, etc.) + Unknown

- 📈 **Balanced Dataset:** Yes

- 🖼️ **Image Type:** Close-up vehicle front-view, realistic traffic camera angles

- 🔗 **Dataset Link:** **[Dataset](https://www.kaggle.com/datasets/alirezachahardoli/iranian-car-imageclassification)**



---



## 🧠 Model Architecture



A **custom Convolutional Neural Network (CNN)** was designed and trained **from scratch** in PyTorch with the following features:


- **Loss Function** : I used **`CrossEntropyLoss`** , which is a standard choice for multi-class classification problems.
  
- 🧩 **Skip Connections** (inspired by ResNet)

- **OPtimizer :** The model is optimized using **`SGD`**

- 🔁 **ReduceLROnPlateau** scheduler for automatic LR adjustment

- 🔥 High performance:  

  - **Train Accuracy:** ~99%  

  - **Test Accuracy:** ~99%  

  - **Both Loss:** Extremely low

- ⚡️ Inference speed: ~**5.05 ms per 100 images**



---



## 🧪 Training Configuration
- ✅ Dsigned specifically for **Iranian Vehicles** 

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
# Confusion matrix
<img width="784" height="665" alt="Conf-final" src="https://github.com/user-attachments/assets/19ca82bd-ae4c-4b25-8523-846ebf6c4ef0" />

# Sample Prediction
<img width="1263" height="1194" alt="tk" src="https://github.com/user-attachments/assets/93541f38-4a0a-474b-be00-856c960f20ef" />
            


---



# 🛠 How to Run:

## 🔗 Clone the Repository
```bash
git clone https://github.com/AlirezaChahardoli/Iranian-Vehicles-Classification.git
cd Iranian-Vehicles-Classification
```

## installa Requirements

```bash
pip install -r requirements.txt
```

## Run the Gradio app:
```bash
python app.py
```
## For training the model from scratch:
```bash
python main.py
```
