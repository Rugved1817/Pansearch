# PDF Form Filling Guide

This directory contains scripts to fill PDF forms with dynamic data while preserving logos and watermarks.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pypdf reportlab
```

## Scripts Overview

### 1. `fill_pdf.py` - Form Field Filling
Use this when your PDF has **fillable form fields** (AcroForm).

**Usage:**
```python
from fill_pdf import fill_pdf_form, inspect_pdf_fields

# First, inspect the PDF to see available fields
fields = inspect_pdf_fields("sample.pdf")
print(fields)

# Fill with data
data = {
    "Name": "John Doe",
    "Address": "123 Main St, Mumbai",
    "PAN": "ABCDE1234F"
}
fill_pdf_form("sample.pdf", "output.pdf", data)
```

**Command line:**
```bash
python fill_pdf.py
```

### 2. `fill_pdf_overlay.py` - Text Overlay
Use this when your PDF **doesn't have form fields** and you need to overlay text at specific coordinates.

**Usage:**
```python
from fill_pdf_overlay import fill_pdf_simple

# Define where to place text (coordinates in points)
field_positions = {
    "name": (100, 700),    # x=100, y=700 from bottom-left
    "address": (100, 680),
    "pan": (100, 660),
}

# Data to fill
data = {
    "name": "John Doe",
    "address": "123 Main St",
    "pan": "ABCDE1234F"
}

fill_pdf_simple("sample.pdf", "output.pdf", data, field_positions)
```

**Command line:**
```bash
python fill_pdf_overlay.py
```

### 3. `fill_pdf_unified.py` - Automatic Detection
This script automatically detects whether your PDF has form fields and uses the appropriate method.

**Usage:**
```python
from fill_pdf_unified import fill_pdf

data = {
    "name": "John Doe",
    "address": "123 Main St",
    "pan": "ABCDE1234F"
}

# Auto-detect method
fill_pdf("sample.pdf", "output.pdf", data)

# Or specify method explicitly
fill_pdf("sample.pdf", "output.pdf", data, method="form")      # Force form fields
fill_pdf("sample.pdf", "output.pdf", data, method="overlay")  # Force overlay
```

**Command line:**
```bash
python fill_pdf_unified.py
```

## Finding Coordinates for Overlay Method

If your PDF doesn't have form fields, you need to find the coordinates where you want to place text:

1. Open your PDF in a PDF viewer (Adobe Reader, etc.)
2. Note the positions where you want text
3. Convert to points: 1 inch = 72 points
4. Remember: (0,0) is at the **bottom-left** corner

Example:
- If text should be 1 inch from left and 2 inches from top on an A4 page (8.5" x 11"):
  - x = 1 * 72 = 72 points
  - y = (11 - 2) * 72 = 648 points (from bottom)

## Example: Filling Multiple PDFs

```python
from fill_pdf_unified import fill_pdf

# List of people to process
people = [
    {"name": "John Doe", "address": "123 Main St", "pan": "ABCDE1234F"},
    {"name": "Jane Smith", "address": "456 Park Ave", "pan": "FGHIJ5678K"},
]

for i, person in enumerate(people):
    fill_pdf(
        "sample.pdf",
        f"output_{i+1}.pdf",
        person
    )
```

## Preserving Logo and Watermark

All scripts preserve existing PDF content (logos, watermarks, images) by:
- **Form field method**: Only modifying form field values, not touching other content
- **Overlay method**: Adding text as a new layer on top, preserving underlying content

## Troubleshooting

### "No form fields found"
- Your PDF doesn't have fillable form fields
- Use `fill_pdf_overlay.py` instead
- Or add form fields using a PDF editor (Adobe Acrobat, etc.)

### "Text appears in wrong position"
- Adjust the coordinates in `field_positions`
- Remember: (0,0) is bottom-left, not top-left
- Use a PDF viewer to measure exact positions

### "Import errors"
- Make sure you've installed dependencies: `pip install pypdf reportlab`

### "PDF is encrypted"
- The script will try to decrypt with empty password
- If that fails, you'll need to provide the password (modify the script)

## Notes

- The scripts preserve all existing PDF content including logos and watermarks
- Form field method is preferred when available (cleaner, more reliable)
- Overlay method works for any PDF but requires coordinate specification
- Both methods create new PDF files (original is not modified)

