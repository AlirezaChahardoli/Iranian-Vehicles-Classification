import torch
from torch import nn
import torchmetrics
from tqdm.auto import tqdm
from timeit import default_timer as timer

"""
training and testing loops.
"""

def train_loop(model=nn.Module,
               data=torch.utils.data.DataLoader,
               loss_fn=nn.Module,
               optimizer=torch.optim.Optimizer,
               scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
               accuracy_fn=torchmetrics,
               device=torch.device):
    torch.cuda.manual_seed(42)
    start_time_train=timer()
    train_acc,train_loss=0,0
    model.train()
    for image,label in data:
        image,label=image.to(device),label.to(device)
        y_pred_prob=model(image)
        loss=loss_fn(y_pred_prob,label)
        train_loss+=loss
        train_acc+=accuracy_fn(label,torch.argmax(y_pred_prob,dim=1))
        #train_acc+=acc


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step(train_loss)
    train_acc/=len(data)
    train_loss/=len(data)

    end_time_train=timer()
    current_lr=scheduler.optimizer.param_groups[0]['lr']


    print(f"\033[92m Learinig rate:{current_lr:.6f}\nTrain_loss:{train_loss:.4f}%\n Train_acc:{train_acc:.4f}%\nDevice is on {device}\nTime passed for training:{end_time_train-start_time_train:.2f}s\033[0m")

    return train_acc,train_loss