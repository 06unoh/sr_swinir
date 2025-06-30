from utils.transforms import get_pre_tf, get_tensor_tf, get_train_tf

from torchvision import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

class ImageDataset(Dataset):
  def __init__(self, filelist, hr_size=256, scale_factor=4, mode='train'):
    self.files=[]
    broken_count=5
    for file in filelist:
      try:
        with Image.open(file) as img:
          width,height=img.size
          if width>=hr_size and height>=hr_size:
            self.files.append(file)
      except Exception as e:
        print(f"broken img {file} : {e}")
        broken_count-=1
        if broken_count==0:
          raise RuntimeError(f'Too many broken img')

    self.hr_size=hr_size
    self.lr_size=self.hr_size//scale_factor

    if mode=='train':
      self.pre_tf=get_train_tf(self.hr_size)
    else:
      self.pre_tf=get_pre_tf(self.hr_size)

    self.tensor_tf=get_tensor_tf()

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    try:
      img=Image.open(self.files[idx]).convert('RGB')
      hr_img=self.pre_tf(img)
      lr_img=F.resize(hr_img, (self.lr_size, self.lr_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
      hr_img=self.tensor_tf(hr_img)
      lr_img=self.tensor_tf(lr_img)
      return lr_img, hr_img
    except Exception as e:
      print(f"❗ Error loading image: {self.files[idx]} – {e}")
      # 무작위 다른 이미지 리턴 (안전 fallback)
