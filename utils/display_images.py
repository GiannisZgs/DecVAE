"""
Simple script to display an image in notebook output.

Usage from notebook:
    !python utils/display_images.py <image_path>
    
Example:
    !python utils/display_images.py figures/fig_2_vowels_results/visualization.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    if len(sys.argv) < 2:
        print("Usage: python display_images.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    try:
        # Read and display the image
        img = mpimg.imread(str(image_path))
        
        # Create figure with appropriate size
        dpi = 100
        height, width = img.shape[:2]
        figsize = width / dpi, height / dpi
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Save to stdout as base64 so it displays in notebook
        plt.savefig(sys.stdout.buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
