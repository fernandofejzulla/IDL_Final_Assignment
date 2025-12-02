import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import os
# import torchvision.utils as vutils # No longer needed
import matplotlib.pyplot as plt

# --- 1. User's Data Loading Function (Unchanged logic) ---
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

    # Note: Keras 'sigmoid' output expects [0, 1]. 
    # If using 'tanh' in generator, use scale=True to get [-1, 1].
    if scale:
        X = (X - 127.5) * 2
    else:
        X = X / 255.0

    return X

# --- 2. PyTorch Model Definitions ---

class Generator(nn.Module):
    """
    Mimics build_deconv_net
    Input: Latent Vector
    Structure: Dense -> Reshape -> 4x Conv2DTranspose -> 1x Conv2D
    """
    def __init__(self, latent_dim=100, filters=128):
        super(Generator, self).__init__()
        
        # 1. Dense Layer: Keras logic was 4*4*64
        self.fc = nn.Linear(latent_dim, 4 * 4 * 64)
        
        # 2. Upsampling layers
        # Keras used 'same' padding with kernel 3, stride 2.
        # In PyTorch: kernel=3, stride=2, padding=1, output_padding=1 exactly doubles dimensions.
        self.upsample_blocks = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(64, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # Block 4: 32x32 -> 64x64
            nn.ConvTranspose2d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True)
        )
        
        # 3. Final Convolutional Layer (matches Keras "dec_output")
        # Keras: Conv2D(filters=3, kernel=3, padding='same')
        # PyTorch: Conv2d(in, 3, kernel=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(filters, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid() # Matches activation_out='sigmoid'

    def forward(self, z):
        # Dense -> (Batch, 1024)
        x = self.fc(z)
        # Reshape -> (Batch, 64, 4, 4) -- Note: Channels First for PyTorch
        x = x.view(-1, 64, 4, 4)
        x = self.upsample_blocks(x)
        x = self.final_conv(x)
        return self.sigmoid(x)

class Discriminator(nn.Module):
    """
    Mimics build_conv_net
    Input: (3, 64, 64) Image
    Structure: Conv2D -> 4x Downsampling Conv2D -> Flatten -> Dense
    """
    def __init__(self, in_shape=(3, 64, 64), filters=128):
        super(Discriminator, self).__init__()
        
        # 1. Initial Input Conv
        # Keras: Conv2D(filters, kernel=3, stride=2, padding='same')
        # PyTorch: kernel=3, stride=2, padding=1 maps 64 -> 32
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_shape[0], filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # 2. Downsampling Loop (4 layers)
        layers = []
        # Input to loop is 32x32.
        # Layer 1: 32 -> 16
        # Layer 2: 16 -> 8
        # Layer 3: 8 -> 4
        # Layer 4: 4 -> 2
        for _ in range(4):
            layers.append(nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(True))
            
        self.downsample_blocks = nn.Sequential(*layers)
        
        # 3. Output
        # At 2x2 with 128 filters, flattened size is 128 * 2 * 2 = 512
        self.flatten_dim = filters * 2 * 2
        self.fc = nn.Linear(self.flatten_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.initial_conv(img)
        x = self.downsample_blocks(x)
        x = torch.flatten(x, start_dim=1) # Flatten logic
        x = self.fc(x)
        return self.sigmoid(x)

# --- 3. Training Setup ---

def train():
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    img_dir = "data/s4561341/cats/" # <--- UPDATE THIS
    latent_dim = 100
    batch_size = 64
    lr = 0.0002
    epochs = 50

    # 1. Load Data
    # Important: Keras 'sigmoid' output implies [0,1] data. So scale=False (default).
    X_train = load_real_samples(img_dir, scale=False, img_size=64)
    
    if len(X_train) == 0:
        print("No images loaded. Exiting.")
        return

    # Convert Numpy (N, 64, 64, 3) -> PyTorch (N, 3, 64, 64)
    tensor_x = torch.from_numpy(X_train).permute(0, 3, 1, 2)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Init Models
    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)

    # 3. Optimization
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr)
    optimizerD = optim.Adam(netD.parameters(), lr=lr)

    # Track losses
    G_losses = []
    D_losses = []

    print("Starting training...")

    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        for i, (imgs,) in enumerate(dataloader):
            imgs = imgs.to(device)
            current_batch_size = imgs.size(0)

            # Labels
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizerD.zero_grad()

            # Real loss
            outputs = netD(imgs)
            d_loss_real = criterion(outputs, real_labels)

            # Fake loss
            z = torch.randn(current_batch_size, latent_dim).to(device)
            fake_imgs = netG(z)
            outputs = netD(fake_imgs.detach()) # Detach to stop gradient to G
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizerG.zero_grad()

            # Generator wants Discriminator to think images are real
            outputs = netD(fake_imgs)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizerG.step()
            
            # Accumulate
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        # Average loss per epoch
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        D_losses.append(avg_d_loss)
        G_losses.append(avg_g_loss)

        print(f"[Epoch {epoch}/{epochs}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]")
        
        # Save sample image occasionally
        if epoch % 10 == 0:
            # vutils.save_image(fake_imgs.data[:16], f"epoch_{epoch}.png", nrow=4)
            
            # Using Matplotlib to save a 4x4 grid
            with torch.no_grad():
                # Get first 16 images
                samples = fake_imgs.data[:16].cpu()
                # Permute dimensions: (N, C, H, W) -> (N, H, W, C) for Matplotlib
                samples = samples.permute(0, 2, 3, 1).numpy()
                
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx, ax in enumerate(axes.flat):
                    if idx < len(samples):
                        ax.imshow(samples[idx])
                    ax.axis('off')
                
                plt.suptitle(f"Epoch {epoch}")
                plt.tight_layout()
                plt.savefig(f"epoch_{epoch}.png")
                plt.close(fig)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    print("Loss plot saved to loss_plot.png")

if __name__ == "__main__":
    train()