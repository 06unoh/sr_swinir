import torchvision.transforms as transforms

def get_train_tf(hr_size):
    return transforms.Compose([
        transforms.RandomCrop(hr_size),
        transforms.ColorJitter(0.1, 0.1, 0.1, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((90, 90))
        # transforms.RandomChoice([
        #     transforms.RandomRotation((0, 0)),
        #     transforms.RandomRotation((90, 90)),
        #     transforms.RandomRotation((180, 180)),
        #     transforms.RandomRotation((270, 270))
        # ])
    ])  
    
def get_pre_tf(hr_size):
    return transforms.Compose([
        transforms.CenterCrop(hr_size)
    ])

    
def get_tensor_tf():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.486, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    