import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# --- 1. User's Data Loading Function ---
def load_real_samples(image_dir, scale=False, img_size=64, limit=20000):
    images = []
    count = 0
    
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} not found.")
        return np.array([])

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(image_dir, filename)
            img = cv2.imread(path)
            if img is None: continue 

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            count += 1
            if count >= limit: break

    X = np.array(images, dtype=np.float32)

    if scale:
        X = (X - 127.5) * 2
    else:
        X = X / 255.0

    return X

# --- 2. Utils: Weight Initialization ---
def weights_init(m):
    """
    Custom weights initialization called on netG and netD.
    Standard for DCGAN: mean=0, std=0.02
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- 3. PyTorch Model Definitions ---

class Generator(nn.Module):
    def __init__(self, latent_dim=100, filters=128):
        super(Generator, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 4 * 4 * 64)
        
        self.upsample_blocks = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(64, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            
            # Block 4: 32x32 -> 64x64
            nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True)
        )
        
        self.final_conv = nn.Conv2d(filters, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 4, 4)
        x = self.upsample_blocks(x)
        x = self.final_conv(x)
        return self.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self, in_shape=(3, 64, 64), filters=128):
        super(Discriminator, self).__init__()
        
        # 1. Initial Input Conv (No BatchNorm on first layer usually)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_shape[0], filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. Downsampling Loop
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        self.downsample_blocks = nn.Sequential(*layers)
        
        self.flatten_dim = filters * 2 * 2
        self.fc = nn.Linear(self.flatten_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.initial_conv(img)
        x = self.downsample_blocks(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

# --- 4. Training Setup ---

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    img_dir = "/data/s4561341/cats/" 
    
    # Configurations to iterate over
    latent_dims_list = [256,512]
    batch_sizes_list = [64, 128]
    
    lrG = 0.0002
    lrD = 0.00002
    epochs = 100

    # 1. Load Data ONCE
    print("Loading data...")
    X_train = load_real_samples(img_dir, scale=False, img_size=64)
    
    if len(X_train) == 0:
        print("No images loaded. Exiting.")
        return

    # Convert Numpy -> PyTorch Tensor
    tensor_x = torch.from_numpy(X_train).permute(0, 3, 1, 2)
    full_dataset = TensorDataset(tensor_x)

    print(f"Data loaded. Total images: {len(full_dataset)}")

    # 2. Loop through configurations
    for batch_size in batch_sizes_list:
        for latent_dim in latent_dims_list:
            
            config_name = f"bs{batch_size}_ld{latent_dim}"
            print(f"\n--- Starting training for config: {config_name} ---")

            # Create specific dataloader for this batch size
            dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

            # Init Models for this config
            netG = Generator(latent_dim=latent_dim).to(device)
            netD = Discriminator().to(device)
            
            # Apply weights initialization
            netG.apply(weights_init)
            netD.apply(weights_init)

            # Optimization
            criterion = nn.BCELoss()
            optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999))
            optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.5, 0.999))

            # Track losses for this specific config
            G_losses = []
            D_losses = []
            
            # Fixed noise for consistent visualization (optional, but good for last epoch)
            fixed_noise = torch.randn(16, latent_dim).to(device)

            for epoch in range(epochs):
                epoch_d_loss = 0.0
                epoch_g_loss = 0.0
                
                for i, (imgs,) in enumerate(dataloader):
                    imgs = imgs.to(device)
                    current_batch_size = imgs.size(0)

                    # Labels
                    real_labels = torch.ones(current_batch_size, 1).to(device)
                    fake_labels = torch.zeros(current_batch_size, 1).to(device)

                    # --- Train Discriminator ---
                    optimizerD.zero_grad()

                    outputs = netD(imgs)
                    d_loss_real = criterion(outputs, real_labels)

                    z = torch.randn(current_batch_size, latent_dim).to(device)
                    fake_imgs = netG(z)
                    outputs = netD(fake_imgs.detach()) 
                    d_loss_fake = criterion(outputs, fake_labels)

                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    optimizerD.step()

                    # --- Train Generator ---
                    optimizerG.zero_grad()

                    outputs = netD(fake_imgs)
                    g_loss = criterion(outputs, real_labels)

                    g_loss.backward()
                    optimizerG.step()
                    
                    epoch_d_loss += d_loss.item()
                    epoch_g_loss += g_loss.item()

                avg_d_loss = epoch_d_loss / len(dataloader)
                avg_g_loss = epoch_g_loss / len(dataloader)
                D_losses.append(avg_d_loss)
                G_losses.append(avg_g_loss)

                # Log every 10 epochs
                if epoch % 10 == 0:
                    print(f"[{config_name}] [Epoch {epoch}/{epochs}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]")
                
                # --- SAVE ONLY LAST EPOCH IMAGES ---
                if epoch == epochs - 1:
                    print(f"Saving final images for {config_name}...")
                    with torch.no_grad():
                        # Use fixed noise for the final save
                        fake_final = netG(fixed_noise).detach().cpu()
                        
                        # Prepare plot
                        samples = fake_final.permute(0, 2, 3, 1).numpy()
                        
                        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                        for idx, ax in enumerate(axes.flat):
                            if idx < len(samples):
                                ax.imshow(samples[idx])
                            ax.axis('off')
                        
                        plt.suptitle(f"Final Epoch: {config_name}")
                        plt.tight_layout()
                        plt.savefig(f"results/generated_{config_name}.png")
                        plt.close(fig)

            # --- SAVE LEARNING CURVES FOR THIS CONFIG ---
            plt.figure(figsize=(10, 5))
            plt.title(f"Losses: {config_name}")
            plt.plot(G_losses, label="Generator")
            plt.plot(D_losses, label="Discriminator")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"loss_plot_{config_name}.png")
            plt.close()
            print(f"Saved loss plot for {config_name}")

if __name__ == "__main__":
    train()