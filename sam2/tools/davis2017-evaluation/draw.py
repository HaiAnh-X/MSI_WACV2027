import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
image_dir = "/home/ids/lnguyen-23/haianh/data/DAVIS/JPEGImages/480p"     # ảnh gốc DAVIS
mask_dir  = "/home/ids/lnguyen-23/haianh/sam2/davis2017-evaluation/results0.999"                # mask output của bạn
out_dir   = "vis_output"                # folder save visualization

os.makedirs(out_dir, exist_ok=True)

alpha = 0.5  # độ trong suốt mask

# màu cho object (BGR)
COLOR = (0, 0, 255)  # đỏ

# =========================
# LOAD SEQUENCES
# =========================
seq_names = sorted(os.listdir(image_dir))

for seq in tqdm(seq_names):

    seq_img_path = os.path.join(image_dir, seq)
    seq_mask_path = os.path.join(mask_dir, seq)
    seq_out_path = os.path.join(out_dir, seq)

    os.makedirs(seq_out_path, exist_ok=True)

    if not os.path.isdir(seq_img_path):
        continue

    frames = sorted(os.listdir(seq_img_path))

    for frame_name in frames:
        img_path = os.path.join(seq_img_path, frame_name)

        mask_path = os.path.join(seq_mask_path, frame_name.replace(".jpg", ".png"))

        if not os.path.exists(mask_path):
            continue

        # =========================
        # READ IMAGE + MASK
        # =========================
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # resize mask nếu cần
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # binary mask
        mask_bin = mask > 0

        # =========================
        # OVERLAY
        # =========================
        overlay = img.copy()

        overlay[mask_bin] = (
            (1 - alpha) * img[mask_bin] + alpha * np.array(COLOR)
        )

        overlay = overlay.astype(np.uint8)

        # =========================
        # SAVE
        # =========================
        save_path = os.path.join(seq_out_path, frame_name)
        cv2.imwrite(save_path, overlay)

print("Done: visualization saved.")