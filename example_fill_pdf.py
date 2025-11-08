"""
Example script showing how to fill PDF forms with dynamic data.

This script demonstrates how to use the fill_pdf module to:
1. Inspect PDF form fields
2. Fill PDF forms with dynamic data
3. Preserve logos and watermarks
"""

from fill_pdf import fill_pdf_form, inspect_pdf_fields
from pathlib import Path


def example_fill_pdf():
    """
    Example: Fill a PDF form with sample data.
    """
    # Path to your PDF file
    input_pdf = "sample.pdf"
    output_pdf = "sample_filled.pdf"
    
    # Check if PDF exists
    if not Path(input_pdf).exists():
        print(f"Error: {input_pdf} not found!")
        return
    
    # Step 1: Inspect the PDF to see what fields are available
    print("Step 1: Inspecting PDF form fields...")
    print("-" * 50)
    fields = inspect_pdf_fields(input_pdf)
    
    if not fields:
        print("No form fields found in the PDF.")
        print("The PDF may not have fillable form fields.")
        return
    
    print(f"Found {len(fields)} form field(s):")
    for field_name, field_info in fields.items():
        print(f"  - {field_name}: {field_info.get('type', 'Unknown')} "
              f"(current: '{field_info.get('current_value', 'Empty')}')")
    print()
    
    # Step 2: Prepare your data
    # Replace these with your actual data
    data = {
        # Example: Map field names to values
        # "Name": "John Doe",
        # "Address": "123 Main Street, Mumbai, Maharashtra 400001",
        # "PAN": "ABCDE1234F",
        # "Date": "2024-01-15",
        # "Phone": "9876543210",
    }
    
    # Auto-populate based on field names (you can customize this)
    for field_name in fields.keys():
        field_lower = field_name.lower()
        if "name" in field_lower:
            if field_name not in data:
                data[field_name] = "John Doe"
        elif "address" in field_lower:
            if field_name not in data:
                data[field_name] = "123 Main Street, Mumbai, Maharashtra 400001"
        elif "pan" in field_lower:
            if field_name not in data:
                data[field_name] = "ABCDE1234F"
        elif "date" in field_lower:
            if field_name not in data:
                data[field_name] = "2024-01-15"
        elif "phone" in field_lower or "mobile" in field_lower:
            if field_name not in data:
                data[field_name] = "9876543210"
    
    if not data:
        print("No data to fill. Please add data to the 'data' dictionary.")
        return
    
    # Step 3: Fill the PDF
    print("Step 2: Filling PDF with data...")
    print("-" * 50)
    print("Data to fill:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    print()
    
    success = fill_pdf_form(
        input_pdf_path=input_pdf,
        output_pdf_path=output_pdf,
        data=data,
        preserve_appearance=True  # Preserves logo and watermark
    )
    
    if success:
        print(f"\n✓ Success! Filled PDF saved to: {output_pdf}")
        print("  Note: Logo and watermark are preserved.")
    else:
        print("\n✗ Failed to fill PDF. Check the error messages above.")


def fill_pdf_with_custom_data(name: str, address: str, pan: str = "", date: str = "", phone: str = ""):
    """
    Example function to fill PDF with custom data.
    
    Args:
        name: Person's name
        address: Full address
        pan: PAN number (optional)
        date: Date (optional)
        phone: Phone number (optional)
    """
    input_pdf = "sample.pdf"
    output_pdf = f"filled_{name.replace(' ', '_')}.pdf"
    
    # First, inspect to get actual field names
    fields = inspect_pdf_fields(input_pdf)
    
    # Map data to field names (case-insensitive matching)
    data = {}
    for field_name in fields.keys():
        field_lower = field_name.lower()
        if "name" in field_lower:
            data[field_name] = name
        elif "address" in field_lower:
            data[field_name] = address
        elif "pan" in field_lower and pan:
            data[field_name] = pan
        elif "date" in field_lower and date:
            data[field_name] = date
        elif ("phone" in field_lower or "mobile" in field_lower) and phone:
            data[field_name] = phone
    
    if fill_pdf_form(input_pdf, output_pdf, data, preserve_appearance=True):
        print(f"✓ Created filled PDF: {output_pdf}")
        return output_pdf
    else:
        print("✗ Failed to create filled PDF")
        return None


if __name__ == "__main__":
    # Run the example
    example_fill_pdf()
    
    # Or use the custom function:
    # fill_pdf_with_custom_data(
    #     name="Jane Smith",
    #     address="456 Park Avenue, New Delhi, Delhi 110001",
    #     pan="FGHIJ5678K",
    #     date="2024-02-20",
    #     phone="9876543210"
    # )

