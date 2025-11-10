#!/usr/bin/env python3
"""
Script to check if Devanagari fonts are available for PDF generation.
Run this to diagnose font issues.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("ERROR: reportlab not installed. Install with: pip install reportlab")
    sys.exit(1)


def check_font_paths():
    """Check which font paths exist."""
    resources_dir = Path(__file__).parent / "resources"
    
    font_paths = [
        # Resources folder
        resources_dir / "NotoSansDevanagari-Regular.ttf",
        resources_dir / "NotoSansDevanagari-Bold.ttf",
        resources_dir / "fonts" / "NotoSansDevanagari-Regular.ttf",
        resources_dir / "fonts" / "NotoSansDevanagari-Bold.ttf",
        # Windows
        Path("C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf"),
        Path("C:/Windows/Fonts/NotoSansDevanagari-Bold.ttf"),
        Path("C:/Windows/Fonts/mangal.ttf"),
        Path("C:/Windows/Fonts/MANGAL.TTF"),
        Path("C:/Windows/Fonts/mangalb.ttf"),
        Path("C:/Windows/Fonts/MANGALB.TTF"),
        Path("C:/Windows/Fonts/ARIALUNI.TTF"),
    ]
    
    print("Checking for Devanagari fonts...")
    print("=" * 60)
    
    found_fonts = {"regular": [], "bold": []}
    
    for font_path in font_paths:
        if font_path.exists():
            font_name = font_path.name
            is_bold = "bold" in font_name.lower() or "b" in font_name.lower() and "mangalb" in font_name.lower()
            font_type = "bold" if is_bold else "regular"
            found_fonts[font_type].append(str(font_path))
            print(f"✓ FOUND ({font_type:7}): {font_path}")
        else:
            print(f"✗ NOT FOUND: {font_path}")
    
    print("=" * 60)
    
    if found_fonts["regular"]:
        print(f"\n✓ Regular font found: {found_fonts['regular'][0]}")
        # Try to register it
        try:
            pdfmetrics.registerFont(TTFont("Test-Devanagari", found_fonts["regular"][0]))
            print("✓ Font registration successful!")
        except Exception as e:
            print(f"✗ Font registration failed: {e}")
    else:
        print("\n✗ No regular Devanagari font found!")
        print("\nTo fix this:")
        print("1. Download Noto Sans Devanagari from: https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari")
        print(f"2. Place NotoSansDevanagari-Regular.ttf in: {resources_dir}")
        print("3. Or install it system-wide (Windows: copy to C:/Windows/Fonts/)")
    
    if found_fonts["bold"]:
        print(f"\n✓ Bold font found: {found_fonts['bold'][0]}")
    else:
        print("\n⚠ No bold Devanagari font found (optional, but recommended)")
    
    # Test Unicode rendering
    print("\n" + "=" * 60)
    print("Testing Unicode rendering...")
    try:
        test_text = "नमस्ते"  # "Namaste" in Devanagari
        print(f"Test text: {test_text}")
        print("✓ Unicode text handling appears correct")
    except Exception as e:
        print(f"✗ Unicode handling issue: {e}")
    
    return found_fonts


if __name__ == "__main__":
    check_font_paths()

