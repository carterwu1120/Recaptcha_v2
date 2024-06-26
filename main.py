# https://github.com/AarohiSingla/Image-Classification-Using-Vision-transformer/blob/main/image_classifier_from_scratch.ipynb
# https://hyper.ai/tutorials/24825
import pandas as pd
import torch
from torchvision import transforms
from torch import nn
from torchinfo import summary
from tqdm.auto import tqdm
import argparse
import os
import datetime
# from going_modular.going_modular import engine
from model_run import *
from data_loader import *
from model import *
# To check out our ViT model's loss curves, we can use the plot_loss_curves function from helper_functions.py
from helper_functions import plot_loss_curves

device = "cuda" if torch.cuda.is_available() else "cpu"
# python main.py -mode=1 -load_dir=train_ViT2024-06-02-23-29-03
# Set seeds
def set_seeds(seed: int=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default=0, type=int)
    parser.add_argument("-img_size", default=72, type=int)
    parser.add_argument("-batch_size", default=8, type=int)
    parser.add_argument("-patch_size", default=6, type=int)
    parser.add_argument("-embedding_dim", default=64, type=int)
    parser.add_argument("-mlp_size", default=128, type=int)
    parser.add_argument("-num_heads", default=4, type=int)
    parser.add_argument("-attn_dropout", default=0, type=float)
    parser.add_argument("-mlp_dropout", default=0.1, type=float)
    parser.add_argument("-lr", default=1e-3, type=float)
    parser.add_argument("-betas", default=(0.9, 0.999), type=any)
    parser.add_argument("-weight_decay", default=0.001, type=float)
    parser.add_argument("-data_dir", default="data\\2classes_data", type=str)
    parser.add_argument("-load_dir", default="", type=str)
    parser.add_argument("-save_dir", default="logs", type=str)
    parser.add_argument("-model_name", default="ViT.pt", type=str)
    parser.add_argument("-num_classes", default=2, type=int)
    parser.add_argument("-epochs", default=200, type=int)
    
    return parser

def save_train(save_dir):
    log = None
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
    model_file_name = "ViT"
    save_dir = '{}/train_{}/'.format(save_dir, model_file_name + timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_file = os.path.join(save_dir, model_file_name + '.pt')
    log_file = os.path.join(save_dir, '_log.txt')
    log = open(log_file, 'w')

    return log, save_dir, model_file
def load_model(load_dir, save_dir):
    
    save_dir = '{}/{}/'.format(save_dir, load_dir)

    model_file = os.path.join(save_dir, 'ViT.pt')    

    checkpoint = torch.load(model_file, map_location = device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint['acc']

    log_file = os.path.join(save_dir, '_log.txt')
    log = open(log_file, 'a')

    return log, save_dir, model_file, start_epoch, best_val_acc
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
        
    # Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
    model = ViT(img_size=args.img_size, 
                patch_size=args.patch_size, 
                embedding_dim=args.embedding_dim,
                mlp_size=args.mlp_size,
                num_heads=args.num_heads,
                attn_dropout=args.attn_dropout,
                mlp_dropout=args.mlp_dropout,
                num_classes=args.num_classes).to(device)

    # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper 
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                lr=args.lr, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=args.betas, # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                weight_decay=args.weight_decay) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    if args.load_dir:
        log, save_dir, model_file, start_epoch, best_acc = load_model(args.load_dir, args.save_dir)
    else:
        log, save_dir, model_file = save_train(args.save_dir)
        start_epoch = 0  
        best_acc = 0 
    if args.mode == 0:
    # Create transform pipeline manually
        manual_transforms = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  
        print(f"Manually created transforms: {manual_transforms}") 
        train_dir = os.path.join(args.data_dir, "train")
        validate_dir = os.path.join(args.data_dir, "validate")
        
        train_dataloader, validate_dataloader, class_names = create_training_dataloaders(
            train_dir=train_dir,
            validate_dir=validate_dir,
            transform=manual_transforms, 
            batch_size=args.batch_size
        )
        # Get a batch of images
        image_batch, label_batch = next(iter(train_dataloader))

        # Get a single image from the batch
        image, label = image_batch[0], label_batch[0]

        # 2. Print shape of original image tensor and get the image dimensions
        print(f"Image tensor shape: {image.shape}")
        height, width = image.shape[1], image.shape[2]

        # 3. Get image tensor and add batch dimension
        x = image.unsqueeze(0)
        print(f"Input image with batch dimension shape: {x.shape}")

        set_seeds()
        # 4. Create patch embedding layer
        patch_embedding_layer = PatchEmbedding(in_channels=3,
                                            patch_size=args.patch_size,
                                            embedding_dim=args.embedding_dim)

        # 5. Pass image through patch embedding layer
        patch_embedding = patch_embedding_layer(x)
        print(f"Patching embedding shape: {patch_embedding.shape}")



        # 6. Create class token embedding
        batch_size = patch_embedding.shape[0]
        embedding_dimension = patch_embedding.shape[-1]
        class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                                requires_grad=True) # make sure it's learnable
        print(f"Class token embedding shape: {class_token.shape}")

        # 7. Prepend class token embedding to patch embedding
        patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
        print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

        # 8. Create position embedding
        number_of_patches = int((height * width) / args.patch_size**2)
        position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                        requires_grad=True) # make sure it's learnable


        # 9. Add position embedding to patch embedding with class token
        patch_and_position_embedding = patch_embedding_class_token + position_embedding
        print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")
        #patch_and_position_embedding

        print(patch_embedding_class_token)  #1 is added in the beginning of each

        transformer_encoder_block = TransformerEncoderBlock(embedding_dim=args.embedding_dim, 
                                                            num_heads=args.num_heads).to(device)
        summary(model=transformer_encoder_block,
            input_size=(args.batch_size, number_of_patches, args.embedding_dim), # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )

        # Save args
        print(args, file=log,end="\n-\n")
        log.flush()

        # Setup the loss function for multi-class classification
        # loss_fn = torch.nn.BCELoss()
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the model and save the training results to a dictionary
        results = {"train_loss": [],
                "train_acc": [],
                "train_auc": [],
                "test_loss": [],
                "test_acc": [],
                "test_auc": [],
        }
        best_epoch = start_epoch
        for epoch in tqdm(range(start_epoch, args.epochs)):
            train_loss, train_acc = train_Vit(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            test_loss, test_acc = validate_Vit(model=model,
            dataloader=validate_dataloader,
            loss_fn=loss_fn,
            device=device)
            if (test_acc >= best_acc):
                best_acc = test_acc
                best_epoch = epoch
                print(f"[INFO] Saving model to: {model_file}")
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'acc': test_acc
                    }, model_file)
                
                # Print out what's happening
                print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f} | ",
                file = log)
                log.flush()

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
        plot_loss_curves(results, args.save_dir)
    elif args.mode == 1:
        manual_transforms = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  
        print(f"Manually created transforms: {manual_transforms}") 
        test_dir = os.path.join("data", "test image")
        test_dataloader = create_testing_dataloaders(
            test_dir=test_dir,
            transform=manual_transforms, 
            batch_size=1
        )
        
        target_list = pd.read_excel("data/testinglabel.xlsx")["class(1 for 1, 0 for 9)"].values.tolist()
        target_list = [1 if target == 0 else 0 for target in target_list]
        
        # _ = load_model(args.load_dir, args.save_dir)
        print("start testing")
        # auc, acc = recaptcha_v2_testing(model=model, 
        #                                 dataloader=test_dataloader, 
        #                                 target_list=target_list, 
        #                                 device=device)
        auc, acc = recaptcha_v2_testing(model=model, 
                                        manual_transforms=manual_transforms, 
                                        target_list=target_list, 
                                        device=device)
        print("auc : ", auc)
        print("acc : ", acc)
