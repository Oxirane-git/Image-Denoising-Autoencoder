import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib parameters for high-quality output
plt.rcParams['figure.dpi'] = 300  # High DPI for crisp output
plt.rcParams['savefig.dpi'] = 300  # High DPI for saved figures
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# Directories
noisy_dir = "images/noisy"
clean_dir = "images/clean"

class DenoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.transform = transform
        self.noisy_images = sorted([f for f in os.listdir(noisy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.clean_images = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create balanced pairs - repeat clean images to match noisy count
        if len(self.noisy_images) > len(self.clean_images):
            # Repeat clean images to match noisy images count
            multiplier = len(self.noisy_images) // len(self.clean_images)
            remainder = len(self.noisy_images) % len(self.clean_images)
            balanced_clean = self.clean_images * multiplier + self.clean_images[:remainder]
            self.pairs = list(zip(self.noisy_images, balanced_clean))
        else:
            # Use random sampling if clean > noisy
            paired_clean = random.sample(self.clean_images, len(self.noisy_images))
            self.pairs = list(zip(self.noisy_images, paired_clean))
            
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        print(f"Created {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.pairs[idx][0])
        clean_path = os.path.join(self.clean_dir, self.pairs[idx][1])
        
        noisy_img = Image.open(noisy_path).convert("L")
        clean_img = Image.open(clean_path).convert("L")
        
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img

# Enhanced transform with better preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256), Image.LANCZOS),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = DenoiseDataset(noisy_dir, clean_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class SimplifiedDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(SimplifiedDenoisingAutoencoder, self).__init__()
        
        # Encoder - much simpler
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x128
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x64
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x32
        
        # Bottleneck - simplified
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder - simplified with correct channel dimensions
        self.upconv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)  # 64x64
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),  # 128 + 128 = 256 from skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 128x128
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),  # 64 + 64 = 128 from skip connection
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # 256x256
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 32 + 32 = 64 from skip connection
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(16, 1, 1)
        
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)          # 32 channels, 256x256
        enc1_pool = self.pool1(enc1) # 32 channels, 128x128
        
        enc2 = self.enc2(enc1_pool)  # 64 channels, 128x128
        enc2_pool = self.pool2(enc2) # 64 channels, 64x64
        
        enc3 = self.enc3(enc2_pool)  # 128 channels, 64x64
        enc3_pool = self.pool3(enc3) # 128 channels, 32x32
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3_pool)  # 128 channels, 32x32
        
        # Decoder with skip connections
        up3 = self.upconv3(bottleneck)           # 128 channels, 64x64
        concat3 = torch.cat([up3, enc3], dim=1)  # 128 + 128 = 256 channels
        dec3 = self.dec3(concat3)                # 64 channels, 64x64
        
        up2 = self.upconv2(dec3)                 # 64 channels, 128x128
        concat2 = torch.cat([up2, enc2], dim=1)  # 64 + 64 = 128 channels
        dec2 = self.dec2(concat2)                # 32 channels, 128x128
        
        up1 = self.upconv1(dec2)                 # 32 channels, 256x256
        concat1 = torch.cat([up1, enc1], dim=1)  # 32 + 32 = 64 channels
        dec1 = self.dec1(concat1)                # 16 channels, 256x256
        
        output = torch.sigmoid(self.final_conv(dec1))  # 1 channel, 256x256
        
        return output

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        l1_loss = self.l1(output, target)
        
        # Edge preservation loss
        def gradient_loss(pred, target):
            pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
            target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
            
            return self.mse(pred_grad_x, target_grad_x) + self.mse(pred_grad_y, target_grad_y)
        
        grad_loss = gradient_loss(output, target)
        
        return 0.6 * mse_loss + 0.3 * l1_loss + 0.1 * grad_loss

# Initialize model, loss, and optimizer
model = SimplifiedDenoisingAutoencoder().to(device)
criterion = CombinedLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Training loop
num_epochs = 120
train_losses = []
best_loss = float('inf')

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (noisy, clean) in enumerate(dataloader):
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_denoiser.pth")
        print(f"New best model saved with loss: {best_loss:.4f}")
    
    # Save checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoint_epoch_{epoch+1}.pth')

# Save final model
torch.save(model.state_dict(), "final_denoiser.pth")

# Enhanced plot with better quality
def plot_training_curve():
    """Create high-quality training loss plot"""
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, linewidth=2, color='#2E8B57', label='Training Loss')
    plt.title('Image Denoising Model - Training Loss Curve', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig('training_loss_hq.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Plot the training curve
plot_training_curve()

def denoise_image_enhanced(image_path=None, save_path=None):
    """Enhanced denoising function with better visualization and metrics"""
    model.eval()
    
    if image_path is None:
        img_path = "/home/matrix/sahil_project/images/noisy/Fontfse_Noisec_TR.png"
    else:
        img_path = image_path
    
    # Load and preprocess image
    img = Image.open(img_path).convert("L")
    original_size = img.size
    
    # Transform image
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze().cpu()
    
    # Convert back to PIL Image with original size
    output_np = output.numpy()
    output_np = np.clip(output_np, 0, 1)
    output_img = Image.fromarray((output_np * 255).astype(np.uint8), mode='L')
    output_img = output_img.resize(original_size, Image.LANCZOS)
    
    # Calculate metrics
    img_resized = img.resize((256, 256), Image.LANCZOS)
    img_np = np.array(img_resized) / 255.0
    
    # PSNR calculation
    mse = np.mean((img_np - output_np) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Enhanced visualization with metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Image Denoising Results - Enhanced Visualization', fontsize=16, fontweight='bold')
    
    # Original noisy image
    axes[0, 0].imshow(img, cmap="gray", interpolation='nearest')
    axes[0, 0].set_title("Original Noisy Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Denoised output
    axes[0, 1].imshow(output_img, cmap="gray", interpolation='nearest')
    axes[0, 1].set_title(f"Denoised Output\nPSNR: {psnr:.2f} dB", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference map
    diff = np.array(img_resized) - output_np * 255
    im_diff = axes[0, 2].imshow(diff, cmap="RdBu_r", interpolation='nearest')
    axes[0, 2].set_title("Noise Removed\n(Difference Map)", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[0, 2], shrink=0.6)
    
    # Histograms for comparison
    axes[1, 0].hist(np.array(img).flatten(), bins=50, alpha=0.7, color='red', label='Noisy', density=True)
    axes[1, 0].set_title("Pixel Intensity Distribution\n(Original)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Pixel Intensity")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(np.array(output_img).flatten(), bins=50, alpha=0.7, color='green', label='Denoised', density=True)
    axes[1, 1].set_title("Pixel Intensity Distribution\n(Denoised)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Pixel Intensity")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Quality metrics text
    metrics_text = f"""
    Quality Metrics:
    
    PSNR: {psnr:.2f} dB
    Original Size: {original_size}
    Processing Size: 256x256
    
    Model Architecture:
    - U-Net with Skip Connections
    - Residual Blocks in Bottleneck
    - Combined Loss (MSE + L1 + Gradient)
    
    Noise Reduction: {np.std(img_np):.4f} → {np.std(output_np):.4f}
    """
    
    axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save with high quality
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        output_img.save(save_path.replace('.png', '_denoised.png'))
        print(f"High-quality results saved to {save_path}")
    else:
        plt.savefig('denoising_result_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
        output_img.save('denoised_output_enhanced.png')
        print("High-quality results saved to denoising_result_enhanced.png and denoised_output_enhanced.png")
    
    plt.show()
    
    # Print detailed metrics
    print(f"\n{'='*50}")
    print(f"DENOISING PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
    print(f"Original Image Standard Deviation: {np.std(img_np):.4f}")
    print(f"Denoised Image Standard Deviation: {np.std(output_np):.4f}")
    print(f"Noise Reduction: {((np.std(img_np) - np.std(output_np)) / np.std(img_np) * 100):.1f}%")
    print(f"Original Size: {original_size}")
    print(f"Processing Size: 256x256 pixels")
    print(f"{'='*50}")
    
    return output_img

# Test the enhanced denoising
print("\nTesting enhanced denoising visualization...")
denoise_image_enhanced()