import os
from PIL import Image

# Path to the uploaded grid image in brain directory
grid_img_path = r"C:\Users\Shashwat\.gemini\antigravity-ide\brain\edb074b7-346c-4a20-9c3b-9dd5dfd01a23\media__1779721547220.jpg"
target_dir = r"c:\Users\Shashwat\OneDrive\Desktop\cdss\frontend\public\products"

# Ensure target folder exists
os.makedirs(target_dir, exist_ok=True)

if os.path.exists(grid_img_path):
    print(f"Loading user grid image from: {grid_img_path}")
    img = Image.open(grid_img_path)
    width, height = img.size
    print(f"Grid Dimensions: {width}x{height}")

    # Calculate coordinates for 4 equal quadrants
    mid_x = width // 2
    mid_y = height // 2

    # Quadrant 1: Top-Left (OCR Ingestion Core Scanner)
    print("Cropping Quadrant 1 (Top-Left) -> glucowave.webp")
    q1 = img.crop((0, 0, mid_x, mid_y))
    q1.save(os.path.join(target_dir, "glucowave.webp"), "WEBP", quality=85, method=6)
    print(f"Saved glucowave.webp: {os.path.getsize(os.path.join(target_dir, 'glucowave.webp')) / 1024:.2f} KB")

    # Quadrant 3: Bottom-Left (DDI Node Network Map)
    print("Cropping Quadrant 3 (Bottom-Left) -> touchwave.webp")
    q3 = img.crop((0, mid_y, mid_x, height))
    q3.save(os.path.join(target_dir, "touchwave.webp"), "WEBP", quality=85, method=6)
    print(f"Saved touchwave.webp: {os.path.getsize(os.path.join(target_dir, 'touchwave.webp')) / 1024:.2f} KB")

    # Quadrant 4: Bottom-Right (Symmetrical Clinical Predictor Hub Board)
    print("Cropping Quadrant 4 (Bottom-Right) -> mpvt.webp")
    q4 = img.crop((mid_x, mid_y, width, height))
    q4.save(os.path.join(target_dir, "mpvt.webp"), "WEBP", quality=85, method=6)
    print(f"Saved mpvt.webp: {os.path.getsize(os.path.join(target_dir, 'mpvt.webp')) / 1024:.2f} KB")

else:
    print(f"Grid image not found at: {grid_img_path}")
