import os
import random
random.seed(123)
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()
def create_dataloaders(train_dir, test_dir, transform, batch_size):
    num_workers = NUM_WORKERS
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader, class_names
    
if __name__ == '__main__':
    # Create image size
    IMG_SIZE = 224

    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])           
    print(f"Manually created transforms: {manual_transforms}")

    BATCH_SIZE = 32
    train_dir = r'D:\NTNU\112-2\AI\term_project\Recaptcha_v2\data\train'
    test_dir = r'D:\NTNU\112-2\AI\term_project\Recaptcha_v2\data\test'
    # Create data loaders
    train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, 
    batch_size=BATCH_SIZE
    )
    # Let's visualize a image in order to know if data is loaded properly or not
    # Get a batch of images
    image_batch, label_batch = next(iter(train_dataloader))
    # Get a single image from the batch
    image, label = image_batch[0], label_batch[0]

    # View the batch shapes
    print(image.shape, label)

    # Plot image with matplotlib
    plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(class_names[label])
    # plt.axis(False)
    plt.show()