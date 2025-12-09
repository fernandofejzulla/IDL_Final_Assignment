import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, filters=128):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 4 * 4 * 64)
        self.upsample_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, filters, 3, 2, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True),
            nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True),
            nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True),
            nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True)
        )
        self.final_conv = nn.Conv2d(filters, 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 4, 4)
        x = self.upsample_blocks(x)
        return self.sigmoid(self.final_conv(x))

class Discriminator(nn.Module):
    def __init__(self, in_shape=(3, 64, 64), filters=128):
        super(Discriminator, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_shape[0], filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(filters, filters, 3, 2, 1))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.downsample_blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(filters * 2 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.initial_conv(img)
        x = self.downsample_blocks(x)
        x = torch.flatten(x, start_dim=1)
        return self.sigmoid(self.fc(x))

import json

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists("results"):
        os.makedirs("results")

    latent_dims_list = [128, 256]
    batch_sizes_list = [64, 128]
    
    seeds = [42, 101, 999] 
    
    lrG = 0.0002
    lrD = 0.00002
    epochs = 100

    with open("config.json", "r") as file:
        config = json.load(file)

    X_train = load_real_samples(config["data_folder"], scale=False, img_size=64)
    if len(X_train) == 0:
        print("No images loaded. Exiting.")
        return

    tensor_x = torch.from_numpy(X_train).permute(0, 3, 1, 2)
    full_dataset = TensorDataset(tensor_x)
    print(f"Data loaded. Total images: {len(full_dataset)}")

    experiment_results = {}
    for batch_size in batch_sizes_list:
        for latent_dim in latent_dims_list:
            
            config_name = f"bs{batch_size}_ld{latent_dim}"
            experiment_results[config_name] = {'G': [], 'D': []}
            
            for seed in seeds:
                print(f"\n--- Starting: {config_name} | Seed: {seed} ---")
                set_seed(seed) 

                dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
                netG = Generator(latent_dim=latent_dim).to(device)
                netD = Discriminator().to(device)
                netG.apply(weights_init)
                netD.apply(weights_init)

                criterion = nn.BCELoss()
                optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999))
                optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.5, 0.999))

                seed_G_losses = []
                seed_D_losses = []
                fixed_noise = torch.randn(16, latent_dim).to(device)

                for epoch in range(epochs):
                    epoch_d_loss = 0.0
                    epoch_g_loss = 0.0
                    
                    for i, (imgs,) in enumerate(dataloader):
                        imgs = imgs.to(device)
                        curr_bs = imgs.size(0)
                        real_labels = torch.ones(curr_bs, 1).to(device)
                        fake_labels = torch.zeros(curr_bs, 1).to(device)

                        # Train D
                        optimizerD.zero_grad()
                        outputs = netD(imgs)
                        d_loss_real = criterion(outputs, real_labels)

                        z = torch.randn(curr_bs, latent_dim).to(device)
                        fake_imgs = netG(z)
                        outputs = netD(fake_imgs.detach()) 
                        d_loss_fake = criterion(outputs, fake_labels)

                        d_loss = d_loss_real + d_loss_fake
                        d_loss.backward()
                        optimizerD.step()

                        # Train G
                        optimizerG.zero_grad()
                        outputs = netD(fake_imgs)
                        g_loss = criterion(outputs, real_labels)
                        g_loss.backward()
                        optimizerG.step()
                        
                        epoch_d_loss += d_loss.item()
                        epoch_g_loss += g_loss.item()

                    avg_d = epoch_d_loss / len(dataloader)
                    avg_g = epoch_g_loss / len(dataloader)
                    seed_D_losses.append(avg_d)
                    seed_G_losses.append(avg_g)

                    if epoch % 10 == 0:
                        print(f"[{config_name}] [Seed {seed}] [Epoch {epoch}] D: {avg_d:.4f} G: {avg_g:.4f}")

                experiment_results[config_name]['G'].append(seed_G_losses)
                experiment_results[config_name]['D'].append(seed_D_losses)

                with torch.no_grad():
                    fake_final = netG(fixed_noise).detach().cpu()
                    samples = fake_final.permute(0, 2, 3, 1).numpy()
                    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                    for idx, ax in enumerate(axes.flat):
                        if idx < len(samples): ax.imshow(samples[idx])
                        ax.axis('off')
                    plt.suptitle(f"Config: {config_name} | Seed: {seed}")
                    plt.tight_layout()
                    plt.savefig(f"results/gen_{config_name}_seed{seed}.png")
                    plt.close(fig)

            g_data = np.array(experiment_results[config_name]['G']) 
            d_data = np.array(experiment_results[config_name]['D'])
            
            g_mean = np.mean(g_data, axis=0)
            g_std = np.std(g_data, axis=0)
            d_mean = np.mean(d_data, axis=0)
            d_std = np.std(d_data, axis=0)
            
            epochs_range = range(epochs)

            plt.figure(figsize=(10, 6))
            
            plt.plot(epochs_range, g_mean, label="Generator Mean", color="blue")
            plt.fill_between(epochs_range, g_mean - g_std, g_mean + g_std, color="blue", alpha=0.2)
            
            plt.plot(epochs_range, d_mean, label="Discriminator Mean", color="orange")
            plt.fill_between(epochs_range, d_mean - d_std, d_mean + d_std, color="orange", alpha=0.2)
            
            plt.title(f"Aggregated Performance (3 Seeds): {config_name}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss (Log Scale)")
            plt.legend()
            
            plt.yscale('log')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            plt.savefig(f"results/loss_agg_log_{config_name}.png")
            plt.close()
            print(f"Saved results/loss_agg_log_{config_name}.png")

if __name__ == "__main__":
    train()