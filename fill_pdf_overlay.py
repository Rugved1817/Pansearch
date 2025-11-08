"""
Script to fill PDF by overlaying text at specific coordinates.
Use this when the PDF doesn't have fillable form fields.

This approach preserves logo and watermark by only adding text layers.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, ArrayObject, FloatObject, IndirectObject
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import io
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    print(f"Warning: Required libraries not available. Install with: pip install pypdf reportlab")
    print(f"Error: {e}")

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def overlay_text_on_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    text_data: Dict[str, Tuple[float, float, str]],
    page_size: Tuple[float, float] = A4
) -> bool:
    """
    Overlay text on a PDF at specific coordinates.
    
    Args:
        input_pdf_path: Path to input PDF
        output_pdf_path: Path to output PDF
        text_data: Dict mapping page numbers (1-indexed) to list of (x, y, text) tuples
                  Example: {1: [(100, 700, "John Doe"), (100, 680, "123 Main St")]}
        page_size: Page size tuple (width, height) in points
    
    Returns:
        True if successful
    """
    if not REPORTLAB_AVAILABLE:
        print("Error: reportlab is required. Install with: pip install reportlab")
        return False
    
    try:
        # Read the original PDF
        reader = PdfReader(input_pdf_path)
        
        if reader.is_encrypted:
            if not reader.decrypt(""):
                print("Error: PDF is encrypted and requires a password.")
                return False
        
        # Create a writer
        writer = PdfWriter()
        
        # Process each page
        for page_num, page in enumerate(reader.pages, start=1):
            # Get page dimensions
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)
            
            # Create a PDF in memory for the overlay
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(page_width, page_height))
            
            # Add text for this page if specified
            if page_num in text_data:
                for x, y, text in text_data[page_num]:
                    # PDF coordinates: (0,0) is bottom-left
                    # y coordinate needs to be from bottom
                    can.drawString(x, page_height - y, str(text))
            
            can.save()
            
            # Move to the beginning of the StringIO buffer
            packet.seek(0)
            overlay_pdf = PdfReader(packet)
            overlay_page = overlay_pdf.pages[0]
            
            # Merge the overlay with the original page
            page.merge_page(overlay_page)
            
            # Add the merged page to writer
            writer.add_page(page)
        
        # Write the output PDF
        with open(output_pdf_path, "wb") as output_file:
            writer.write(output_file)
        
        return True
        
    except Exception as e:
        print(f"Error overlaying text: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def fill_pdf_simple(
    input_pdf_path: str,
    output_pdf_path: str,
    data: Dict[str, str],
    field_positions: Optional[Dict[str, Tuple[float, float]]] = None
) -> bool:
    """
    Simple function to fill PDF with data at predefined positions.
    
    Args:
        input_pdf_path: Input PDF path
        output_pdf_path: Output PDF path
        data: Dictionary of field names to values
        field_positions: Optional dict mapping field names to (x, y) coordinates
                        If None, uses default positions
    
    Returns:
        True if successful
    """
    # Default positions (you need to adjust these based on your PDF layout)
    # Coordinates are in points (72 points = 1 inch)
    # (x, y) where (0,0) is bottom-left corner
    default_positions = {
        "name": (100, 700),      # Adjust based on your PDF
        "address": (100, 680),
        "pan": (100, 660),
        "date": (100, 640),
        "phone": (100, 620),
    }
    
    positions = field_positions or default_positions
    
    # Prepare text data for overlay
    text_data = {1: []}  # Page 1
    
    for field_name, value in data.items():
        field_lower = field_name.lower()
        
        # Find matching position
        for pos_name, (x, y) in positions.items():
            if pos_name in field_lower or field_lower in pos_name:
                text_data[1].append((x, y, str(value)))
                break
    
    return overlay_text_on_pdf(input_pdf_path, output_pdf_path, text_data)


def main():
    """
    Example usage - you need to customize the coordinates for your PDF.
    """
    input_pdf = "sample.pdf"
    output_pdf = "sample_filled.pdf"
    
    if not Path(input_pdf).exists():
        print(f"Error: {input_pdf} not found!")
        return
    
    # Example: Define where to place text on the PDF
    # You need to find the coordinates by opening the PDF in a viewer
    # and noting the positions where you want text
    field_positions = {
        "name": (100, 700),      # x=100, y=700 from bottom-left
        "address": (100, 680),
        "pan": (100, 660),
        "date": (100, 640),
        "phone": (100, 620),
    }
    
    # Data to fill
    data = {
        "name": "John Doe",
        "address": "123 Main Street, Mumbai, Maharashtra 400001",
        "pan": "ABCDE1234F",
        "date": "2024-01-15",
        "phone": "9876543210",
    }
    
    print("Filling PDF with overlay text...")
    print("Note: You may need to adjust coordinates in the script.")
    print()
    
    success = fill_pdf_simple(input_pdf, output_pdf, data, field_positions)
    
    if success:
        print(f"✓ Success! Filled PDF saved to: {output_pdf}")
    else:
        print("✗ Failed to fill PDF.")


if __name__ == "__main__":
    main()

