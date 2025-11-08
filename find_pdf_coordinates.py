"""
Helper script to find coordinates for text placement in PDF.

This script helps you determine where to place text by showing
page dimensions and providing a coordinate reference.
"""

from pathlib import Path

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("Error: pypdf not installed. Install with: pip install pypdf")


def analyze_pdf(pdf_path: str):
    """
    Analyze PDF and show page dimensions and coordinate system info.
    """
    if not PYPDF_AVAILABLE:
        return
    
    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found!")
        return
    
    reader = PdfReader(pdf_path)
    
    if reader.is_encrypted:
        if not reader.decrypt(""):
            print("Error: PDF is encrypted.")
            return
    
    print(f"PDF Analysis: {pdf_path}")
    print("=" * 60)
    print(f"Number of pages: {len(reader.pages)}")
    print()
    
    for i, page in enumerate(reader.pages, 1):
        mediabox = page.mediabox
        width = float(mediabox.width)
        height = float(mediabox.height)
        
        # Convert to inches
        width_in = width / 72
        height_in = height / 72
        
        print(f"Page {i}:")
        print(f"  Dimensions: {width:.1f} x {height:.1f} points")
        print(f"  Dimensions: {width_in:.2f} x {height_in:.2f} inches")
        print(f"  Coordinate system: (0,0) at bottom-left")
        print(f"  Top-right corner: ({width:.1f}, {height:.1f})")
        print()
        
        # Show some reference points
        print("  Reference points (from bottom-left):")
        print(f"    Bottom-left: (0, 0)")
        print(f"    Bottom-right: ({width:.1f}, 0)")
        print(f"    Top-left: (0, {height:.1f})")
        print(f"    Top-right: ({width:.1f}, {height:.1f})")
        print(f"    Center: ({width/2:.1f}, {height/2:.1f})")
        print()
        
        # Common positions (1 inch margins)
        margin = 72  # 1 inch = 72 points
        print("  Common positions (1 inch margins):")
        print(f"    Top-left text area: ({margin}, {height - margin})")
        print(f"    Top-center: ({width/2:.1f}, {height - margin})")
        print(f"    Top-right text area: ({width - margin}, {height - margin})")
        print()


def print_coordinate_guide():
    """
    Print a guide for finding coordinates.
    """
    print("\n" + "=" * 60)
    print("HOW TO FIND COORDINATES FOR YOUR PDF")
    print("=" * 60)
    print("""
Method 1: Using a PDF Viewer
-----------------------------
1. Open your PDF in Adobe Reader or similar
2. Note where you want to place text (e.g., "2 inches from left, 1 inch from top")
3. Convert to points: 1 inch = 72 points
4. Remember: Y coordinate is from BOTTOM, not top!

Example:
- If you want text 2" from left and 1" from top on an 8.5" x 11" page:
  - x = 2 * 72 = 144 points
  - y = (11 - 1) * 72 = 720 points (from bottom)

Method 2: Trial and Error
--------------------------
1. Start with estimated coordinates
2. Run the fill script
3. Check the output PDF
4. Adjust coordinates and repeat

Method 3: Use PDF Editor
-------------------------
1. Open PDF in Adobe Acrobat or similar
2. Add form fields at desired locations
3. Use fill_pdf.py (form field method) instead
4. This is easier and more reliable!
""")


if __name__ == "__main__":
    pdf_path = "sample.pdf"
    
    if Path(pdf_path).exists():
        analyze_pdf(pdf_path)
    else:
        print(f"Error: {pdf_path} not found!")
        print("Usage: python find_pdf_coordinates.py")
        print("Make sure sample.pdf is in the current directory.")
    
    print_coordinate_guide()

