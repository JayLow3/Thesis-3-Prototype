
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

st.title("Forest Change Detection System")
st.write(
    "Satellite-Based Detection of Deforestation and "
    "Reforestation In The Philippines Using Image Recognition"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = CBR(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


class UNet3(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = CBR(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


class UNet4(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


model_file = st.file_uploader(
    "Upload Model",
    type=["pth"]
)

model = None
detected_model = None

if model_file is not None:
    try:
        checkpoint = torch.load(
            model_file,
            map_location=device,
            weights_only=True
        )

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "The uploaded file does not contain a valid state dictionary."
            )

        if any(key.startswith("module.") for key in checkpoint.keys()):
            checkpoint = {
                key.removeprefix("module."): value
                for key, value in checkpoint.items()
            }

        if "up4.weight" in checkpoint:
            model = UNet4().to(device)
            detected_model = "4-Block U-Net"
        elif "up3.weight" in checkpoint:
            model = UNet3().to(device)
            detected_model = "3-Block U-Net"
        else:
            model = UNet().to(device)
            detected_model = "2-Block U-Net"

        model.load_state_dict(checkpoint)
        model.eval()

        st.success(f"Successfully loaded {detected_model}")
        st.caption(f"Running on: {device.upper()}")

    except Exception as e:
        model = None
        detected_model = None
        st.error(f"Failed to load model: {e}")


metrics_file = st.file_uploader(
    "Upload Training Metrics",
    type=["npz"]
)

if metrics_file is not None:
    st.subheader("Training Metrics")

    try:
        with np.load(metrics_file, allow_pickle=False) as metrics:

            def pick(*names):
                for name in names:
                    if name in metrics.files:
                        return metrics[name].copy()
                return None

            train_losses = pick(
                "train_losses", "train_loss", "loss_train"
            )
            val_losses = pick(
                "val_losses", "val_loss", "loss_val"
            )
            train_ious = pick(
                "train_ious", "train_iou", "iou_train", "train_IOU"
            )
            val_ious = pick(
                "val_ious", "val_iou", "iou_val", "val_IOU"
            )

        if train_losses is not None and val_losses is not None:
            fig1, ax1 = plt.subplots()

            ax1.plot(
                np.arange(1, len(train_losses) + 1),
                train_losses,
                label="Train Loss"
            )
            ax1.plot(
                np.arange(1, len(val_losses) + 1),
                val_losses,
                label="Validation Loss"
            )

            ax1.set_title("Loss Curve")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.warning("Loss data not found.")

        if train_ious is not None and val_ious is not None:
            fig2, ax2 = plt.subplots()

            ax2.plot(
                np.arange(1, len(train_ious) + 1),
                train_ious,
                label="Train IoU"
            )
            ax2.plot(
                np.arange(1, len(val_ious) + 1),
                val_ious,
                label="Validation IoU"
            )

            ax2.set_title("IoU Curve")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("IoU")
            ax2.legend()
            ax2.grid(True)

            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.warning("IoU data not found.")

    except Exception as e:
        st.error(f"Failed to read training metrics: {e}")


img1 = st.file_uploader(
    "Upload earlier image",
    type=["jpg", "jpeg", "png"],
    key="earlier_image"
)

img2 = st.file_uploader(
    "Upload later image",
    type=["jpg", "jpeg", "png"],
    key="later_image"
)


def preprocess(image):
    image = image.convert("RGB")
    image_np = np.array(image)

    h, w, _ = image_np.shape

    ch, cw = 256, 256

    startx = max(0, w // 2 - cw // 2)
    starty = max(0, h // 2 - ch // 2)

    cropped = image_np[
        starty:starty + ch,
        startx:startx + cw
    ]

    if cropped.shape[0] != 256 or cropped.shape[1] != 256:
        cropped = cv2.copyMakeBorder(
            cropped,
            0,
            max(0, 256 - cropped.shape[0]),
            0,
            max(0, 256 - cropped.shape[1]),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    tensor = (
        transforms.ToTensor()(cropped)
        .unsqueeze(0)
        .to(device)
    )

    return cropped, tensor


def predict(model, tensor):
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))
        pred = pred.squeeze().cpu().numpy()

    return (pred > 0.5).astype(np.uint8)


if st.button("Run"):
    if model is None:
        st.error("Please upload a valid model first.")
        st.stop()

    if img1 is None or img2 is None:
        st.error("Please upload both images.")
        st.stop()

    try:
        img1_c, t1 = preprocess(Image.open(img1))
        img2_c, t2 = preprocess(Image.open(img2))

        with st.spinner("Processing satellite images..."):
            mask1 = predict(model, t1)
            mask2 = predict(model, t2)

        forest1 = np.sum(mask1)
        forest2 = np.sum(mask2)

        total = mask1.size

        r1 = (forest1 / total) * 100
        r2 = (forest2 / total) * 100

        change = r2 - r1

        if change > 0:
            trend = "Reforestation"
        elif change < 0:
            trend = "Deforestation"
        else:
            trend = "No Change"

        st.subheader("Results")
        st.write(f"Model used: **{detected_model}**")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Earlier Forest Cover", f"{r1:.2f}%")

        with col2:
            st.metric("Later Forest Cover", f"{r2:.2f}%")

        st.metric(
            "Forest Cover Change",
            f"{change:+.2f} percentage points"
        )

        st.write(f"Detected Trend: **{trend}**")

        fig, ax = plt.subplots(2, 3, figsize=(12, 8))

        ax[0, 0].imshow(img1_c)
        ax[0, 0].set_title("Earlier Image")

        ax[0, 1].imshow(mask1, cmap="gray")
        ax[0, 1].set_title("Earlier Mask")

        ax[0, 2].imshow(img1_c)
        ax[0, 2].imshow(
            mask1,
            cmap="Greens",
            alpha=0.4,
            vmin=0,
            vmax=1
        )
        ax[0, 2].set_title("Earlier Overlay")

        ax[1, 0].imshow(img2_c)
        ax[1, 0].set_title("Later Image")

        ax[1, 1].imshow(mask2, cmap="gray")
        ax[1, 1].set_title("Later Mask")

        ax[1, 2].imshow(img2_c)
        ax[1, 2].imshow(
            mask2,
            cmap="Greens",
            alpha=0.4,
            vmin=0,
            vmax=1
        )
        ax[1, 2].set_title("Later Overlay")

        for a in ax.flatten():
            a.axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
