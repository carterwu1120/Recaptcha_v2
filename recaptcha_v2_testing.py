
from model import *
def load_model(load_dir, save_dir):
    
    save_dir = '{}/{}/'.format(save_dir, load_dir)

    model_file = os.path.join(save_dir, 'ViT.pt')    

    # store data by dictionary
    # dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'loss', 'auc'])
    checkpoint = torch.load(model_file, map_location = device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    #best_val_loss = checkpoint['loss']
    best_val_auc = checkpoint['acc']

    log_file = os.path.join(save_dir, '_log.txt')
    log = open(log_file, 'a')

    return log, save_dir, model_file, start_epoch, best_val_auc

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dir = "train_ViT2024-06-01-15-11-11"
    model = ViT(img_size=args.img_size, 
            patch_size=args.patch_size, 
            embedding_dim=args.embedding_dim,
            mlp_size=args.mlp_size,
            num_heads=args.num_heads,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            num_classes=len(class_names)).to(device)