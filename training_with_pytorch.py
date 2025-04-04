import os
import torch
import torch.nn as nn
import torch.utils.data as tdata
from dataloader import MyDataset  # Ensure this file is in your Python path
import torchio as tio

# -------------------------------------------------------------------------
# 3D U-Net definition (accepting 3 input channels, 1 output channel)
# -------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=1, init_n_feat=32):
        super().__init__()
        n_feat = init_n_feat

        # Encoder
        self.encoder1 = self._block(ch_in, n_feat)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder2 = self._block(n_feat, n_feat * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder3 = self._block(n_feat * 2, n_feat * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder4 = self._block(n_feat * 4, n_feat * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        # Bottleneck
        self.bottleneck = self._block(n_feat * 8, n_feat * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(n_feat * 16, n_feat * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(n_feat * 8 + n_feat * 8, n_feat * 8)
        self.upconv3 = nn.ConvTranspose3d(n_feat * 8, n_feat * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(n_feat * 4 + n_feat * 4, n_feat * 4)
        self.upconv2 = nn.ConvTranspose3d(n_feat * 4, n_feat * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(n_feat * 2 + n_feat * 2, n_feat * 2)
        self.upconv1 = nn.ConvTranspose3d(n_feat * 2, n_feat, kernel_size=2, stride=2)
        self.decoder1 = self._block(n_feat + n_feat, n_feat)

        self.conv = nn.Conv3d(n_feat, ch_out, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        out = self.conv(d1)
        return torch.sigmoid(out)

# -------------------------------------------------------------------------
# Dice + BCE combined loss
# -------------------------------------------------------------------------
def loss_dice_bce(y_pred, y_true, eps=1e-6):
    numerator = torch.sum(y_true * y_pred, dim=(2, 3, 4)) * 2
    denominator = torch.sum(y_true, dim=(2, 3, 4)) + torch.sum(y_pred, dim=(2, 3, 4)) + eps
    dice_loss = torch.mean(1.0 - (numerator / denominator))
    bce_loss = nn.BCELoss()(y_pred, y_true)
    return dice_loss, bce_loss

# -------------------------------------------------------------------------
# Main training script
# -------------------------------------------------------------------------
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = r"C:\DISSERTATION\Data for Project\Data_by_modality\VOXEL_SIZED_FOLDER"
    print("Initializing datasets...")
    train_set = MyDataset(data_dir, 'train')
    print(f"Datasets initialized. Train size: {len(train_set)}")

    train_loader = tdata.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    model = UNet(ch_in=3, ch_out=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    result_path = "./result"
    os.makedirs(result_path, exist_ok=True)
    step_count = 0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            step_count += 1
            print(f"Step {step_count} | Epoch {epoch+1} - Step {batch_idx+1}/{len(train_loader)} | Processing Patient ID: ")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            dice_loss, bce_loss = loss_dice_bce(model(images), labels)
            loss = 0.5 * dice_loss + 0.5 * bce_loss
            loss.backward()
            optimizer.step()
            print(f"Step {step_count} | Epoch {epoch+1} - Step {batch_idx+1}/{len(train_loader)} | Dice Loss: {dice_loss.item():.5f} | BCE Loss: {bce_loss.item():.5f}")

        checkpoint_file = os.path.join(result_path, f"unet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Saved model checkpoint to {checkpoint_file}")

    print("Training complete.")

if __name__ == "__main__":
    main()
