import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# =========================
# UI HEADER
# =========================

st.title("Forest Change Detection System")
st.write("Satellite-Based Detection of Deforestation and Reforestation In The Philippines Using Image Recognition")

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODEL ARCHITECTURES
# =========================

# -------- 2 BLOCK U-NET --------
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

        # Encoder
        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = CBR(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        # Final
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))

        # Decoder
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


# -------- 4 BLOCK U-NET --------
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

        # Encoder
        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = CBR(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        # Final
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


# =========================
# MODEL UPLOAD
# =========================

model_file = st.file_uploader("Upload Model", type=["pth"])

model = None

if model_file:

    # Load checkpoint first
    checkpoint = torch.load(model_file, map_location=device)

    # AUTO-DETECT MODEL TYPE
    if "up4.weight" in checkpoint:
        model = UNet4().to(device)
        detected_model = "4-Layer U-Net"
    else:
        model = UNet().to(device)
        detected_model = "2-Layer U-Net"

    # Load weights
    model.load_state_dict(checkpoint)

    model.eval()


# =========================
# METRICS UPLOAD (.NPZ)
# =========================

metrics_file = st.file_uploader(
    "Upload Training Metrics",
    type=["npz"]
)

if metrics_file is not None:

    st.subheader("Training Metrics")

    metrics = np.load(metrics_file, allow_pickle=True)

    def pick(*names):
        for n in names:
            if n in metrics.files:
                return metrics[n]
        return None

    train_losses = pick(
        "train_losses",
        "train_loss",
        "loss_train"
    )

    val_losses = pick(
        "val_losses",
        "val_loss",
        "loss_val"
    )

    train_ious = pick(
        "train_ious",
        "train_iou",
        "iou_train",
        "train_IOU"
    )

    val_ious = pick(
        "val_ious",
        "val_iou",
        "iou_val",
        "val_IOU"
    )

    # ===== LOSS CURVE =====
    if train_losses is not None and val_losses is not None:

        epochs = np.arange(len(train_losses))

        fig1, ax1 = plt.subplots()

        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.plot(epochs, val_losses, label="Validation Loss")

        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1.legend()
        ax1.grid(True)

        st.pyplot(fig1)

    else:
        st.warning("Loss data not found")

    # ===== IOU CURVE =====
    if train_ious is not None and val_ious is not None:

        epochs = np.arange(len(train_ious))

        fig2, ax2 = plt.subplots()

        ax2.plot(epochs, train_ious, label="Train IoU")
        ax2.plot(epochs, val_ious, label="Validation IoU")

        ax2.set_title("IoU Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU")

        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig2)

    else:
        st.warning("IoU data not found")


# =========================
# IMAGE UPLOAD
# =========================

img1 = st.file_uploader(
    "Upload earlier image",
    type=["jpg", "jpeg", "png"]
)

img2 = st.file_uploader(
    "Upload later image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess(image):

    image = image.convert("RGB")

    image_np = np.array(image)

    h, w, _ = image_np.shape

    ch, cw = 256, 256

    startx = max(0, w // 2 - cw // 2)
    starty = max(0, h // 2 - ch // 2)

    cropped = image_np[starty:starty + ch, startx:startx + cw]

    # Pad if smaller
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

    tensor = transforms.ToTensor()(cropped).unsqueeze(0).to(device)

    return cropped, tensor


# =========================
# PREDICTION FUNCTION
# =========================

def predict(model, tensor):

    with torch.no_grad():

        pred = torch.sigmoid(
            model(tensor)
        ).squeeze().cpu().numpy()

    return (pred > 0.5).astype(np.uint8)


# =========================
# RUN ANALYSIS
# =========================

if st.button("Run"):

    if model is None:
        st.error("Please upload a model first")
        st.stop()

    if img1 is None or img2 is None:
        st.error("Please upload both images")
        st.stop()

    # Preprocess
    img1_c, t1 = preprocess(Image.open(img1))
    img2_c, t2 = preprocess(Image.open(img2))

    # Predict
    mask1 = predict(model, t1)
    mask2 = predict(model, t2)

    # Forest %
    forest1 = np.sum(mask1)
    forest2 = np.sum(mask2)

    total = mask1.size

    r1 = (forest1 / total) * 100
    r2 = (forest2 / total) * 100

    change = r2 - r1
    abs_change = abs(change)

    # Severity
    if abs_change < 5:
        severity = "No Significant Change"

    elif abs_change < 10:
        severity = "Slight"

    elif abs_change < 25:
        severity = "Moderate"

    elif abs_change < 50:
        severity = "Significant"

    else:
        severity = "Severe"

    # Trend
    if change > 0:
        trend = "Reforestation"

    elif change < 0:
        trend = "Deforestation"

    else:
        trend = "No Change"

    # =========================
    # RESULTS
    # =========================

    st.subheader("Results")

    st.write(f"Earlier Forest Cover: {r1:.2f}%")
    st.write(f"Later Forest Cover: {r2:.2f}%")
    st.write(f"Net Change: {change:.2f}%")
    st.write(f"Trend: {trend}")
    st.write(f"Severity Level: {severity}")

    # =========================
    # VISUALIZATION
    # =========================

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # Earlier
    ax[0,0].imshow(img1_c)
    ax[0,0].set_title("Earlier Image")

    ax[0,1].imshow(mask1, cmap="gray")
    ax[0,1].set_title("Earlier Mask")

    ax[0,2].imshow(img1_c)
    ax[0,2].imshow(mask1, cmap="Greens", alpha=0.4)
    ax[0,2].set_title("Earlier Overlay")

    # Later
    ax[1,0].imshow(img2_c)
    ax[1,0].set_title("Later Image")

    ax[1,1].imshow(mask2, cmap="gray")
    ax[1,1].set_title("Later Mask")

    ax[1,2].imshow(img2_c)
    ax[1,2].imshow(mask2, cmap="Greens", alpha=0.4)
    ax[1,2].set_title("Later Overlay")

    # Remove axes
    for a in ax.flatten():
        a.axis("off")

    st.pyplot(fig)