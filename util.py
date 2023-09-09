import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
from skimage import io, color


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None, target_size=(256, 256)):
        self.image_paths = image_paths
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        
        # Resize the image to the target size
        image = image.resize(self.target_size, Image.BILINEAR)
        
        # Convert the image to RGB if it has an alpha channel
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert the image to Lab color space
        lab_image = color.rgb2lab(np.array(image))
        
        # Separate the L channel
        #l_channel = lab_image[:, :, 0]
        
        if self.transform:
            lab_image = self.transform(lab_image) #l_channel = self.transform(l_channel)
        
        return lab_image #l_channel


def he_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def calculate_metrics(generator, dataloader, device):
    generator.eval()  # Set the generator to evaluation mode

    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for real_images in dataloader:
            real_images = real_images.to(device)

            fake_images = torch.cat([real_images[:, 0:1, :, :], generator(real_images[:, 0:1, :, :])], dim=1)
            
            # Convert tensor to numpy array
            real_images = real_images.cpu().numpy().transpose(0, 2, 3, 1)
            fake_images = fake_images.cpu().numpy().transpose(0, 2, 3, 1)

            # Scale the pixel values to the range [0, 255]
            scaled_image1 = np.clip(real_images.astype(float) * 255.0 / real_images.max(), 0, 255).astype(np.uint8)
            scaled_image2 = np.clip(fake_images.astype(float) * 255.0 / fake_images.max(), 0, 255).astype(np.uint8)

            win_size = min(scaled_image1.shape[0], scaled_image1.shape[1]) // 7  # You can adjust the denominator for your needs
            win_size = max(win_size, 7)  # Ensure it's at least 7
            if win_size % 2 == 0:
                win_size += 1  # Ensure it's odd
            for i in range(len(real_images)):
                ssim_value = ssim(real_images[i], fake_images[i], win_size=win_size, 
                                  multichannel=True, channel_axis=-1, 
                                  data_range=scaled_image2.max() - scaled_image2.min())
                psnr_value = psnr(real_images[i], fake_images[i], data_range=255)

                total_ssim += ssim_value
                total_psnr += psnr_value
                num_samples += 1

    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples

    return avg_psnr, avg_ssim

def collect_image_paths(root_folder, extensions=['.jpg', '.png', '.jpeg']):
    image_paths = []
    
    for folder_name, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(folder_name, filename)
                image_paths.append(full_path)
    
    return image_paths