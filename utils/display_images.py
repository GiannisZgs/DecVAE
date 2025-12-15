"""
Simple script to display an image in notebook.

Usage from notebook (recommended):
    from utils.display_images import show_image
    show_image("figures/fig_2_vowels_results/visualization.png")
    
Or use IPython display directly:
    from IPython.display import Image, display
    display(Image(filename="figures/fig_2_vowels_results/visualization.png"))
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display


def show_image(image_path, width=None, debug=False):
    """
    Display an image in a notebook.
    
    Args:
        image_path: Path to the image file (relative or absolute)
        width: Optional width in pixels for display
        debug: If True, print path resolution information
    """
    # Get current working directory
    cwd = Path(os.getcwd())
    
    # Create path object
    img_path = Path(image_path)
    
    # Resolve to absolute path from current working directory
    if not img_path.is_absolute():
        img_path = (cwd / img_path).resolve()
    
    if debug:
        print(f"Current working directory: {cwd}")
        print(f"Input path: {image_path}")
        print(f"Resolved path: {img_path}")
    
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        print(f"Current working directory: {cwd}")
        return
    
    try:
        if width:
            display(Image(filename=str(img_path), width=width))
        else:
            display(Image(filename=str(img_path)))
    except Exception as e:
        print(f"Error loading image: {e}")


def main():
    """Command-line interface - not recommended for notebook use."""
    if len(sys.argv) < 2:
        print("For notebook use, import the function instead:")
        print("  from utils.display_images import show_image")
        print("  show_image('path/to/image.png')")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Just print the path - shell commands can't display images in notebook output
    print(f"Image exists at: {image_path}")
    print(f"\nTo display in notebook, use:")
    print(f"  from IPython.display import Image, display")
    print(f"  display(Image(filename='{image_path}'))")


if __name__ == "__main__":
    main()
