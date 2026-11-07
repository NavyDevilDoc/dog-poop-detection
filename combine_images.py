"""
combine_images.py
Combines two JPG files into a single JPG, stacked vertically.

Usage:
    python combine_images.py image1.jpg image2.jpg output.jpg

Requirements:
    pip install Pillow
"""

import sys
from PIL import Image


def combine_images(path1: str, path2: str, output_path: str) -> None:
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")

    # If widths differ, scale the narrower image up to match the wider one
    if img1.width != img2.width:
        target_width = max(img1.width, img2.width)

        def scale_to_width(img: Image.Image, width: int) -> Image.Image:
            ratio = width / img.width
            new_height = int(img.height * ratio)
            return img.resize((width, new_height), Image.LANCZOS)

        img1 = scale_to_width(img1, target_width)
        img2 = scale_to_width(img2, target_width)

    combined = Image.new("RGB", (img1.width, img1.height + img2.height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (0, img1.height))

    combined.save(output_path, "JPEG", quality=95)
    print(f"Saved combined image to: {output_path}")
    print(f"  Final size: {combined.width} x {combined.height} px")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_images.py image1.jpg image2.jpg output.jpg")
        sys.exit(1)

    _, path1, path2, output_path = sys.argv
    combine_images(path1, path2, output_path)
