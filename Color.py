import torch
import torch.nn as nn
import torch.nn.functional as F
from util import calculate_metrics, he_init, CustomDataset
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.utils
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image








class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        
        self.encode = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=True), # out: 32 x H/2 x W/2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True), # out: 64 x H/4 x W/4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True), # out: 128 x H/8 x W/8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True), # out: 256 x H/16 x W/16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True), # out: 512 x H/32 x W/32
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
        )

        layers = []        
        for i in range(3):
            layers.append(nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)) # out: 512 x H/32 x W/32
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU())

        self.extract = nn.Sequential(*layers)
        
        self.decode = nn.Sequential(
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), # out: 256 x H/16 x W/16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True), # out: 128 x H/8 x W/8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), # out: 64 x H/4 x W/4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True), # out: 32 x H/2 x W/2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1, bias=True) # out: 2 x H x W  
            
        )

        # Apply He Initialization to all layers
        self.apply(he_init)
        
        
    def forward(self, x):
        
        x = self.encode(x)
        x = self.extract(x)
        x = self.decode(x)

        return x



def train_model(net, num_epochs, train_dataloader, validation_dataloader, device, optimizer):

    pixelwise_loss = nn.L1Loss()

    validation_interval = 1

    # set to training mode
    net.train()

    psnr = []
    ssim = []

    tmax = 5 

    lr_scheduler = StepLR(optimizer, step_size=tmax, gamma=0.5)

    print('Training ...')
    for epoch in range(num_epochs):


        for lab_batch in train_dataloader:

            lab_batch = lab_batch.to(device)

            # apply the color net to the luminance component of the Lab images
            # to get the color (ab) components
            predicted_ab_batch = net(lab_batch[:, 0:1, :, :])

            # loss is the L2 error to the actual color (ab) components
            loss = pixelwise_loss(predicted_ab_batch, lab_batch[:, 1:3, :, :])

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

        if (epoch + 1) % validation_interval == 0:
            avg_psnr, avg_ssim = calculate_metrics(net, validation_dataloader, device)
            psnr += [avg_psnr]
            ssim += [avg_ssim]
            net.train()
            print(f"Epoch [{epoch+1}/{num_epochs}] - PSNR: {avg_psnr:.4f} - SSIM: {avg_ssim:.4f}")

        lr_scheduler.step()
    
    return net, psnr, ssim



def generate_colored_image(grayscale_image, cnet, device, target_size):
    
    original_image = Image.open(grayscale_image)
    # Extract width and height
    original_width, original_height = original_image.size
    
    # Define the data transformation for L channel
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert L channel to tensor
        transforms.Lambda(lambda x: x.to(torch.float32))
    ])
    dataset = CustomDataset([grayscale_image], transform=data_transform, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    cnet.eval()
    with torch.no_grad():
        image = next(iter(dataloader)).to(device)
        image = torch.cat([image[:, 0:1, :, :], cnet(image[:, 0:1, :, :])], dim=1)
        rgb_images = []
        for i in range(image.size(0)):
            lab_img = image[i, :, :, :].detach().cpu().numpy()  # Convert tensor to NumPy array
            rgb_img = color.lab2rgb(np.transpose(lab_img, (1, 2, 0)))  # Convert LAB to RGB
            rgb_images.append(rgb_img)

        # Convert the list of RGB images to a tensor
        rgb_images_tensor = torch.FloatTensor(np.transpose(rgb_images, (0, 3, 1, 2)))
        rgb_images_tensor = transforms.functional.resize(
                rgb_images_tensor, (original_height, original_width), antialias=True
            )

        # Normalize the image
        img = np.transpose(torchvision.utils.make_grid(rgb_images_tensor, nrow=5).numpy(), (1, 2, 0))
        min_value = np.min(img)
        max_value = np.max(img)
        norm_img = (img-min_value)/(max_value-min_value)
        
        
        # Plot the RGB images
        fig, ax = plt.subplots(figsize=(6,6), nrows=1, ncols=1)
        ax.imshow(norm_img)
        ax.title.set_text('Generated RGB Images')
        plt.show()