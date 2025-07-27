import torch
import gradio as gr
from PIL import Image
from torch import nn
import torchvision
from torchvision import transforms,datasets,models

from torchvision.transforms import ToTensor
from custom_model import SimpleResNet
from config import DEVICE 




model=SimpleResNet(num_classes=21).to(DEVICE)

model.load_state_dict(torch.load('model_with_info_path_final.pt',map_location=DEVICE))

model.eval()
classes=['206','207','405','Dena','L90','Mazda-vanet','Naisan','Pars','Paykan-Vanet','Pride','Pride_vanet','Quiek',
         'Saina','Tiba','Truck-Benz','Truck-Renault','Unknown','Volvo-FH-FM','Volvo-N10','Volvo-NH','samand']




transform=transforms.Compose([transforms.Resize((224,224)),
                             transforms.ToTensor(),
                             transforms.Normalize((.5),(.5))])

def classify_image(img1):

    model.eval()
    with torch.inference_mode():
        #img1=Image.open(img1).convert("RGB")
        img1=transform(img1).unsqueeze(0).to(DEVICE)
        y_logits=model(img1)
        y_pred=torch.softmax(y_logits,dim=1)#.argmax(dim=1)
        conf,pred_class=torch.max(y_pred,dim=1)

    if conf.item()<0.55: 
        return f"I'm not sure what this is and confidence:{conf.item():.2f}"
    else:
        return f'Car: {classes[pred_class]}   confidence:{conf.item():.2f}'
        
        #img1=Image.open(img1).convert("RGB")
        #img1=transform(img1).unsqueeze(0).to(device)
        #print(img.shape)
       # y_logits=model(img1)
        #y_pred=torch.softmax(y_logits,dim=1).argmax(dim=1)
        

    #confidence = {classes[i]: float(y_pred[i]) for i in range(len(classes))}

    #return classes[pred_class]

interface = gr.Interface(

    fn=classify_image,

    inputs=gr.Image(type="pil"),

    outputs=gr.Label(num_top_classes=21),

    title="Iranian Car Classifier")

interface.launch(share=True)