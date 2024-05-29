# We recommend Python 3.8 or higher, PyTorch 1.11. 0 or higher and transformers v4. 32.0 or higher.
# https://aravinda-gn.medium.com/how-to-split-image-dataset-into-train-validation-and-test-set-5a41c48af332
import os
import random
import shutil
def CopyImgToDest(class_name, source_path):
    data_path = "data"
    # path to destination folders

    train_folder = os.path.join(os.path.join(data_path, 'train'), class_name)
    val_folder = os.path.join(os.path.join(data_path, 'test'), class_name)

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(source_path) if os.path.splitext(filename)[-1] in image_extensions]
    
    # Sets the random seed 
    random.seed(123) 
    # Shuffle the list of image filenames
    random.shuffle(imgs_list)

    # determine the number of images for each set
    train_size = int(len(imgs_list) * 0.8)
    val_size = int(len(imgs_list) * 0.2)
    
    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        elif i < train_size + val_size:
            dest_folder = val_folder
        elif i == train_size + val_size:
            break
        shutil.copy(os.path.join(source_path, f), os.path.join(dest_folder, f))
        
def DataPreprocess():
    # images_type_names = ['1_image', '9_image']
    images_type_names = ['1_image']
    class_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Other', 'Palm', 'Stair']
    # class_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk']
    # class_names = ['Bicycle', 'Bridge', 'Bus']
    for images_type_name in images_type_names:
        class_path = os.path.join("data/", images_type_name)
        for class_name in class_names:
            source_path = os.path.join(class_path,class_name)
            CopyImgToDest(class_name, source_path)

if __name__ == '__main__':

    DataPreprocess()
