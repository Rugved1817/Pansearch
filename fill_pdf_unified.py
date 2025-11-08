"""
Unified PDF filling script that handles both:
1. Fillable form fields (if PDF has AcroForm)
2. Text overlay at coordinates (if PDF doesn't have form fields)

This script automatically detects which method to use and preserves
logo and watermark in both cases.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from fill_pdf import fill_pdf_form, inspect_pdf_fields

try:
    from fill_pdf_overlay import overlay_text_on_pdf, fill_pdf_simple
    OVERLAY_AVAILABLE = True
except ImportError:
    OVERLAY_AVAILABLE = False
    print("Warning: Overlay functionality not available. Install reportlab: pip install reportlab")


def fill_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    data: Dict[str, Any],
    field_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    method: Optional[str] = None
) -> bool:
    """
    Fill PDF with data using the best available method.
    
    Args:
        input_pdf_path: Path to input PDF
        output_pdf_path: Path to output PDF
        data: Dictionary of field names to values
        field_positions: For overlay method, dict of field names to (x, y) coordinates
        method: 'auto', 'form', or 'overlay'. If None, uses 'auto'
    
    Returns:
        True if successful
    """
    if method is None:
        method = "auto"
    
    # Try form fields first if auto or form method
    if method in ("auto", "form"):
        fields = inspect_pdf_fields(input_pdf_path)
        
        if fields:
            print(f"Found {len(fields)} form field(s). Using form field method...")
            # Filter data to only include fields that exist
            form_data = {k: v for k, v in data.items() if k in fields}
            if form_data:
                return fill_pdf_form(input_pdf_path, output_pdf_path, form_data, preserve_appearance=True)
            else:
                print("Warning: No matching form fields found in data.")
                if method == "form":
                    return False
    
    # Fall back to overlay method
    if method in ("auto", "overlay"):
        if not OVERLAY_AVAILABLE:
            print("Error: Overlay method requires reportlab. Install with: pip install reportlab")
            return False
        
        print("Using text overlay method...")
        return fill_pdf_simple(input_pdf_path, output_pdf_path, data, field_positions)
    
    return False


def main():
    """
    Example usage
    """
    input_pdf = "sample.pdf"
    output_pdf = "sample_filled.pdf"
    
    if not Path(input_pdf).exists():
        print(f"Error: {input_pdf} not found!")
        return
    
    # Your data
    data = {
        "name": "John Doe",
        "address": "123 Main Street, Mumbai, Maharashtra 400001",
        "pan": "ABCDE1234F",
        "date": "2024-01-15",
        "phone": "9876543210",
    }
    
    # For overlay method, define positions (adjust based on your PDF)
    # Coordinates in points: (x, y) where (0,0) is bottom-left
    field_positions = {
        "name": (100, 700),
        "address": (100, 680),
        "pan": (100, 660),
        "date": (100, 640),
        "phone": (100, 620),
    }
    
    print("Filling PDF...")
    success = fill_pdf(
        input_pdf,
        output_pdf,
        data,
        field_positions=field_positions,
        method="auto"  # or "form" or "overlay"
    )
    
    if success:
        print(f"\n✓ Success! Filled PDF saved to: {output_pdf}")
        print("  Logo and watermark are preserved.")
    else:
        print("\n✗ Failed to fill PDF.")


if __name__ == "__main__":
    main()

