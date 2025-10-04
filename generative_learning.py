import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from random import *
from ray.tune.schedulers import ASHAScheduler
from ray import tune
import torch.quantization

#torch.quantization.quantize_dynamic(model=model,{nn.Linear},dtype=torch.qint8)


# config = {
#     'nodes_1':tune.sample_from(lambda _:2**random.randint(2,9)),
#     'nodes_2': tune.sample_from(lambda _: 2**random.randint(2,9)),
#     'lr': tune.loguniform(0.0001,0.1),
#     'batch_size': tune.choice([2,4,8,16])
# }

#Pour ouvrir un fichier ZIP
# from io import BytesIO
# from urllib.request import  urlopen
# from zipfile import ZipFile

# adresse_url = 'lien du document'
# with urlopen(adresse_url) as zipresp:
#     with ZipFile(BytesIO(zipresp.read())) as zfile:
#         zfile.extractall('/extraction')


coding_size = 100
batch_size = 32
image_size = 64

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

dataset = datasets.FashionMNIST(root='fashion',download=False,transform=transform)
dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

data_batch, labels_batch = next(iter(dataloader))
grid_img = make_grid(data_batch, nrow=8)
plt.imshow(grid_img.permute(1, 2, 0))

class Generator(nn.Module):
    def __init__(self, coding_size):
        super(Generator, self).__init__()  # Pas d'espace avant __init__
        self.net = nn.Sequential(
            # coding_size -> 1024 channels, 4x4
            nn.ConvTranspose2d(coding_size, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            # 1024 -> 512 channels, 8x8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # Input doit être 1024, pas coding_size
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 512 -> 256 channels, 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Input doit être 512
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 256 -> 128 channels, 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Input doit être 256
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 128 -> 3 channels (RGB), 64x64
            nn.ConvTranspose2d(128, 1, 4, 2, 1),  # Sortie finale en RGB
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

NetG = Generator(coding_size)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.net  = nn.Sequential(
            nn.Conv2d(1,128,4,2,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,1024,4,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024,1,4,1,0),
            nn.Sigmoid()

        )

    def forward(self,x):
        return self.net(x)
        
netD = Discriminator()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1 :
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)


NetG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(NetG.parameters(),lr=0.0001,betas=(0.5,0.999))
optimizerD = torch.optim.Adam(netD.parameters(),lr=0.0001,betas=(0.5,0.999))

real_labels = torch.full((batch_size,),1.,dtype=torch.float)
fake_labels = torch.full((batch_size,),0.,dtype=torch.float)

G_losses = []
D_losses = []
D_real = []
D_fake = []

z = torch.randn((batch_size,100)).view(-1,100,1,1)

test_out_images = []


n_epochs = 5

for epoch in tqdm(range(n_epochs)):
    print(f"Epoch: {epoch}")

    for i,batch in enumerate(dataloader):
        netD.zero_grad()
        real_images = batch[0]
        output = netD(real_images).view(-1)
        errD_real = criterion(output,real_labels)
        D_x = output.mean().item()

        noise = torch.randn((batch_size,
                coding_size))
        noise = torch.randn(batch_size, 100, 1, 1)
        fake_images = NetG(noise)
        output = netD(fake_images).view(-1)
        errD_fake = criterion(output, fake_labels)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        errD.backward(retain_graph=True)
        optimizerD.step()


         # Train Generator to generate better fakes.
        NetG.zero_grad()
        output = netD(fake_images).view(-1)
        errG = criterion(output, real_labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        # Save losses for plotting later.
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_real.append(D_x)
        D_fake.append(D_G_z2)

        test_images = NetG(z).detach()
        test_out_images.append(test_images)

#scheduler = 