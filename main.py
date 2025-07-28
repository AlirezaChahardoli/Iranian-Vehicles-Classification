"""
main.py runner script.
"""
import torch
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import torchvision
from dataset import get_dataloaders
from custom_model import SimpleResNet
from Train_Test_loop import train_loop,test_loop

from config import DEVICE,BATCH_SIZE,SEED,LEARNING_RATE,EPOCHS
import torchmetrics
from torchmetrics import Accuracy

#from torchmetrics import ConfusionMatrix
#from mlxtend.plotting import plot_confusion_matrix



train_dataloader,test_dataloader,class_names,x=get_dataloaders("your_dataset_path",batch_size=BATCH_SIZE)
model=SimpleResNet(num_classes=len(class_names)).to(DEVICE)

loss_fn=nn.CrossEntropyLoss().to(DEVICE)
optimizer=torch.optim.SGD(params=model.parameters(),lr=0.01)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5,verbose=True)
accuracy=Accuracy(task='multiclass',num_classes=len(class_names)).to(DEVICE)


classes=['206','207','405','Dena','L90','Mazda-vanet','Naisan','Pars','Paykan-Vanet','Pride','Pride_vanet','Quiek',
         'Saina','Tiba','Truck-Benz','Truck-Renault','Unknown','Volvo-FH-FM','Volvo-N10','Volvo-NH','samand']


torch.cuda.manual_seed(42)
epochs=EPOCHS
train_accuracies=[]
train_losses=[]
test_accuracies=[]
test_losses=[]
time_start = timer()
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch}/{epochs}")
    train_acc,train_loss=train_loop(model,train_dataloader,loss_fn,optimizer, scheduler,accuracy)#,device=str(next(model.parameters()).DEVICE))
    test_acc,test_loss=test_loop(model,test_dataloader,loss_fn,accuracy, scheduler)#,device=str(next(model.parameters()).DEVICE))

    train_losses.append(train_loss.item())
    train_accuracies.append(train_acc.item())
    test_losses.append(test_loss.item())
    test_accuracies.append(test_acc.item())
    print(f"------------------------")
time_end=timer()
print(f'Total time passed :\033[91m{time_end-time_start:.4f}s\033[0m')


# save model
torch.save(model.state_dict(),'model_with_info_path_final.pt')
print("Final Model saved")
