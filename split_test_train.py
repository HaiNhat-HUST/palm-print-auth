from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm

data_dir = './roi_test'
train_dir = './train'
test_dir = './test'


if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

    
for folder_name in os.listdir(data_dir):
    train_subdir = os.path.join(train_dir, folder_name)
    
    if not os.path.exists(train_subdir):
        os.makedirs(train_subdir)
    

for folder_name in tqdm(os.listdir(data_dir)):
    folder_path = data_dir + "/" + folder_name
    
    for file in os.listdir(folder_path):
        file_path = folder_path + "/" + file
        file_number = file[9:11]
        if int(file_number) <= 6:
            # print(train_dir + "/" + folder_name + "/" + file)
            shutil.copy(file_path, train_dir + "/" + folder_name + "/" + file)
        else:
            shutil.copy(file_path, test_dir + "/" + file)

