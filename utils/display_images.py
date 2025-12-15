"""
Simple script to display an image.

Usage from notebook:
    !python utils/display_images.py <image_path>
    
Example:
    !python utils/display_images.py figures/fig_2_vowels_results/visualization.png
"""

import sys
from pathlib import Path
from PIL import Image


def main():
    if len(sys.argv) < 2:
        print("Usage: python display_images.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    try:
        img = Image.open(image_path)
        img.show()
        print(f"Displayed: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
