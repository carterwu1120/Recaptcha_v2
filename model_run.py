"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import os
from PIL import Image
import numpy as np
from typing import Tuple
from sklearn import metrics
from torchvision import transforms
manual_transforms = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  
def train_Vit(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        print(type(X))
        print(X.size())
        
        # 1. Forward pass
        y_pred = model(X)
   
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        x = X[0].numpy().transpose(1, 2, 0)
    
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def validate_Vit(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module,
                device: torch.device) -> Tuple[float, float]:
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    validate_loss, validate_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            validate_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            validate_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    validate_loss = validate_loss / len(dataloader)
    validate_acc = validate_acc / len(dataloader)
    return validate_loss, validate_acc

# def recaptcha_v2_testing(model: torch.nn.Module, 
#                         manual_transforms, 
#                         target_list: list,
#                         device: torch.device) -> Tuple[float, float]:
#     # Put model in train mode
#     model.eval()
#     img_pred_labels = []
#     pred_labels = []
#     # Setup train loss and train accuracy values
#     auc, acc = 0, 0

#     floder_dir = "data/test image/test image"
#     for image in os.listdir(floder_dir):
#         image = os.path.join(floder_dir, image)
#         image = Image.open(image)
          
#         for i in range(3):
#             for j in range(3):
#                 tmp_image = image.copy()
#                 img = manual_transforms(tmp_image.crop((120*(i), 120*(j), 120*(i+1), 120*(j+1)))).unsqueeze(0)
#                 img_pred = model(img)
#                 img_pred_label = img_pred.argmax(dim=1)
#                 img_pred_labels.append(img_pred_label)
        
#         if img_pred_labels.count(0)/len(img_pred_labels) > 0.9 :
#             pred_labels.append(0)
#         else:
#             pred_labels.append(1)
        
        
#     all_pred = np.array(pred_labels)
#     all_target = np.array(target_list)

#     acc = metrics.accuracy_score(all_target, all_pred)
#     auc = metrics.roc_auc_score(all_target, all_pred)
#     return acc, auc

def recaptcha_v2_testing(model: torch.nn.Module, 
                        manual_transforms, 
                        target_list: list,
                        device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.eval()
    img_pred_labels = []
    pred_labels = []
    
    # Setup train loss and train accuracy values
    auc, acc = 0, 0
    # floder_dirs = ["data/test/1 image", "data/test/9 image"]
    floder_dirs = ["data/test image/test image"]
    for label, floder_dir in enumerate(floder_dirs):
        for image in os.listdir(floder_dir):
            
            image = os.path.join(floder_dir, image)
            image = Image.open(image).convert('RGB')
            
            for i in range(3):
                for j in range(3):
                    tmp_image = image.copy()
                    
                    img = manual_transforms(tmp_image.crop((120*(i), 120*(j), 120*(i+1), 120*(j+1)))).unsqueeze(0)
                    img_pred = model(img)
                    img_pred_label = img_pred.argmax(dim=1)
                    img_pred_labels.append(img_pred_label)
            
            if img_pred_labels.count(0)/len(img_pred_labels) > 0.9 :
                pred_labels.append(0)
            else:
                pred_labels.append(1)
        
        
    all_pred = np.array(pred_labels)
    all_target = np.array(target_list)

    acc = metrics.accuracy_score(all_target, all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    return acc, auc

# def recaptcha_v2_testing(model: torch.nn.Module, 
#                         dataloader: torch.utils.data.DataLoader, 
#                         target_list: list,
#                         device: torch.device) -> Tuple[float, float]:
#     # Put model in train mode
#     model.eval()
#     pred_list = []
    
#     # Setup train loss and train accuracy values
#     auc, acc = 0, 0

#     # Loop through data loader data batches
#     for batch, (X, y) in enumerate(dataloader):

#         X = X.to(device)

#         y_pred = model(X)
#         y_pred_class = y_pred.argmax(dim=1)
       
#         pred_list.append(y_pred_class.cpu().detach().numpy())
    
#         # target_list.append(y[0])
    
#     all_pred = np.array(pred_list)
#     all_target = np.array(target_list)

#     acc = metrics.accuracy_score(all_target, all_pred)
#     auc = metrics.roc_auc_score(all_target, all_pred)
#     return acc, auc
