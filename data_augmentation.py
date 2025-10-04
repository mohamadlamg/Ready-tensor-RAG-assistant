import torch
import torchvision.transforms as tranforms
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2



transform = tranforms([
    tranforms.ToPILImage(),
    tranforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0),
    tranforms.RandomHorizontalFlip(p=0.5),
    tranforms.RandomGrayscale(p=0.2),
    tranforms.RandomRotation(degrees=45),
    tranforms.ToTensor()
])

#dataset = DAtaset(root='data',transforms=transform)
img_num = 0
for _ in range(10):
    for img,label in dataset:
        save_image(img ,'img'+str(img)+'.png')
        img_num += 1