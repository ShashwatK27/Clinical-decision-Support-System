import os
from PIL import Image

src_path = r"C:\Users\Shashwat\.gemini\antigravity-ide\brain\62503bf6-d4e7-429f-80ec-9525c0a94d54\media__1779809754923.jpg"
target_dir = r"c:\Users\Shashwat\OneDrive\Desktop\cdss\frontend\public"

def process_logo():
    if not os.path.exists(src_path):
        print(f"Source file not found at: {src_path}")
        return

    os.makedirs(target_dir, exist_ok=True)
    img = Image.open(src_path).convert("RGBA")
    width, height = img.size
    print(f"Loaded image size: {width}x{height}")

    # Analyze row-by-row brightness to find the gap between the emblem and the text
    row_activity = []
    pixels = img.load()
    
    for y in range(height):
        active_count = 0
        for x in range(width):
            r, g, b, a = pixels[x, y]
            brightness = (r + g + b) / 3.0
            if brightness > 120:
                active_count += 1
        row_activity.append(active_count)
        
    # We want to find the gap below the emblem.
    # The emblem is in the upper part. Let's look for a gap in rows 500 to 700.
    # A gap is defined by a low active pixel count.
    gap_y = None
    min_activity = 999999
    
    # Search in the middle region of the image for the local minimum row activity
    for y in range(500, 680):
        # Look at a window of 5 rows to be robust to single-pixel lines or noise
        window_activity = sum(row_activity[y-2 : y+3])
        if window_activity < min_activity:
            min_activity = window_activity
            gap_y = y

    print(f"Detected emblem/text separating gap at row Y = {gap_y} (window activity: {min_activity})")

    # If detection failed or was outside boundaries, default to 610
    if gap_y is None or gap_y < 500 or gap_y > 700:
        gap_y = 610
        print(f"Fallback to default gap row: {gap_y}")

    # Create transparent emblem-only and full-logo versions
    def create_transparent_variant(crop_box, name_prefix):
        crop_w = crop_box[2] - crop_box[0]
        crop_h = crop_box[3] - crop_box[1]
        
        img_cream = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
        img_white = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
        img_cyan = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
        
        pix_cream = img_cream.load()
        pix_white = img_white.load()
        pix_cyan = img_cyan.load()
        
        for y_offset, y in enumerate(range(crop_box[1], crop_box[3])):
            for x_offset, x in enumerate(range(crop_box[0], crop_box[2])):
                r, g, b, a = pixels[x, y]
                brightness = (r + g + b) / 3.0
                
                low_thresh = 115
                high_thresh = 155
                
                if brightness <= low_thresh:
                    alpha = 0
                elif brightness >= high_thresh:
                    alpha = 255
                else:
                    alpha = int(((brightness - low_thresh) / (high_thresh - low_thresh)) * 255)
                
                if alpha > 0:
                    pix_cream[x_offset, y_offset] = (r, g, b, alpha)
                    pix_white[x_offset, y_offset] = (250, 250, 250, alpha)
                    pix_cyan[x_offset, y_offset] = (47, 216, 213, alpha)
                    
        # Trim transparent borders for emblem-only to make it perfectly centered and tight
        def get_bbox(img_var):
            return img_var.getbbox()
            
        # Save files
        for img_var, color_name in [(img_cream, "cream"), (img_white, "white"), (img_cyan, "cyan")]:
            bbox = get_bbox(img_var)
            if bbox:
                # Crop to tight bounds to maximize display size
                img_tight = img_var.crop(bbox)
                save_path = os.path.join(target_dir, f"{name_prefix}-{color_name}.png")
                img_tight.save(save_path, "PNG")
                
                # If default cream, also save as standard naming prefix
                if color_name == "cream" and name_prefix == "emblem":
                    img_tight.save(os.path.join(target_dir, "logo-emblem.png"), "PNG")
                elif color_name == "cream" and name_prefix == "logo":
                    img_tight.save(os.path.join(target_dir, "logo.png"), "PNG")
                    
    # 1. Process Emblem Only (from y=0 to y=gap_y)
    create_transparent_variant((0, 0, width, gap_y), "emblem")
    
    # 2. Process Full Logo (from y=0 to y=height)
    create_transparent_variant((0, 0, width, height), "logo")

    print("Successfully processed logo assets with new emblem-only versions.")
    print("Files saved in public directory:")
    print(" - logo-emblem.png (Tight graphical icon)")
    print(" - emblem-cyan.png (Branded Cyan icon)")
    print(" - emblem-white.png (Pure White icon)")
    print(" - logo.png (Full text logo)")

if __name__ == "__main__":
    process_logo()
