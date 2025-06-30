
from datasets.imagedataset import ImageDataset

import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def get_dataloader():
    filepath='./datasets/unsplash'
    all_files=sorted(os.listdir(filepath))
    filelist=[os.path.join(filepath, filename) for filename in all_files if filename.lower().endswith('.jpg')]

    train_files, temp_file=train_test_split(filelist, test_size=0.2, random_state=42)
    val_files, test_files=train_test_split(temp_file, test_size=0.5, random_state=42)

    trainset=ImageDataset(train_files, mode='train')
    valset=ImageDataset(val_files, mode='test')
    testset=ImageDataset(test_files, mode='test')

    trainloader=DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    valloader=DataLoader(valset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    testloader=DataLoader(testset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, valloader, testloader