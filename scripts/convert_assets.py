import os
from PIL import Image

# Source generated paths
source_files = {
    "glucowave.webp": r"C:\Users\Shashwat\.gemini\antigravity-ide\brain\edb074b7-346c-4a20-9c3b-9dd5dfd01a23\glucowave_device_1779717172542.png",
    "touchwave.webp": r"C:\Users\Shashwat\.gemini\antigravity-ide\brain\edb074b7-346c-4a20-9c3b-9dd5dfd01a23\touchwave_device_1779717192779.png",
    "mpvt.webp": r"C:\Users\Shashwat\.gemini\antigravity-ide\brain\edb074b7-346c-4a20-9c3b-9dd5dfd01a23\mpvt_device_1779717212291.png"
}

target_dir = r"c:\Users\Shashwat\OneDrive\Desktop\cdss\frontend\public\products"

# Ensure target folder exists
os.makedirs(target_dir, exist_ok=True)

for target_name, src_path in source_files.items():
    if os.path.exists(src_path):
        print(f"Converting {src_path} -> {target_name}")
        img = Image.open(src_path)
        
        # Keep high resolution (e.g. 1000px wide) but compress WebP format to ensure <300KB
        img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        
        out_path = os.path.join(target_dir, target_name)
        img.save(out_path, "WEBP", quality=85, method=6)
        file_size = os.path.getsize(out_path) / 1024
        print(f"Saved: {out_path} ({file_size:.2f} KB)")
    else:
        print(f"Source file not found: {src_path}")
