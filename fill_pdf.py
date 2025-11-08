"""
Script to fill PDF form fields with dynamic data using pypdf.
Preserves logo and watermark by only modifying form fields.

Usage:
    python fill_pdf.py
    
    Or import and use programmatically:
    from fill_pdf import fill_pdf_form, inspect_pdf_fields
    
    # Inspect fields
    fields = inspect_pdf_fields("sample.pdf")
    
    # Fill with data
    data = {"Name": "John Doe", "Address": "123 Main St"}
    fill_pdf_form("sample.pdf", "output.pdf", data)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, TextStringObject, IndirectObject
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("Warning: pypdf not available. Install with: pip install pypdf")


def inspect_pdf_fields(pdf_path: str) -> Dict[str, Any]:
    """
    Inspect all form fields in a PDF file.
    Returns a dictionary with field names and their properties.
    """
    if not PYPDF_AVAILABLE:
        print("Error: pypdf library is required for this function.")
        return {}
    
    reader = PdfReader(pdf_path)
    
    if reader.is_encrypted:
        print("Warning: PDF is encrypted. Attempting to decrypt...")
        # Try empty password
        if not reader.decrypt(""):
            print("Error: PDF is encrypted and requires a password.")
            return {}
    
    fields_info = {}
    
    # Check for AcroForm fields
    try:
        root = reader.trailer.get("/Root", {})
        if "/AcroForm" in root:
            acro_form = root["/AcroForm"]
            if "/Fields" in acro_form:
                fields = acro_form["/Fields"]
                for field_ref in fields:
                    try:
                        field_obj = field_ref.get_object() if hasattr(field_ref, 'get_object') else field_ref
                        
                        # Get field name
                        field_name_obj = field_obj.get("/T")
                        if field_name_obj:
                            if isinstance(field_name_obj, bytes):
                                field_name = field_name_obj.decode('utf-8', errors='ignore')
                            else:
                                field_name = str(field_name_obj)
                        else:
                            continue
                        
                        # Get field type
                        field_type = field_obj.get("/FT", "Unknown")
                        if isinstance(field_type, bytes):
                            field_type = field_type.decode('utf-8', errors='ignore')
                        else:
                            field_type = str(field_type)
                        
                        # Get current value
                        field_value = field_obj.get("/V", "")
                        if isinstance(field_value, bytes):
                            field_value = field_value.decode('utf-8', errors='ignore')
                        elif field_value:
                            field_value = str(field_value)
                        else:
                            field_value = ""
                        
                        fields_info[field_name] = {
                            "type": field_type,
                            "current_value": field_value,
                        }
                    except Exception as e:
                        print(f"Warning: Could not process field: {e}")
                        continue
    except Exception as e:
        print(f"Warning: Could not read AcroForm: {e}")
    
    # Also check for form fields in page annotations
    for page_num, page in enumerate(reader.pages):
        try:
            if "/Annots" in page:
                annotations = page["/Annots"]
                for annot_ref in annotations:
                    try:
                        annot_obj = annot_ref.get_object() if hasattr(annot_ref, 'get_object') else annot_ref
                        if "/Subtype" in annot_obj and annot_obj["/Subtype"] == "/Widget":
                            if "/T" in annot_obj:
                                field_name_obj = annot_obj["/T"]
                                if isinstance(field_name_obj, bytes):
                                    field_name = field_name_obj.decode('utf-8', errors='ignore')
                                else:
                                    field_name = str(field_name_obj)
                                
                                if field_name and field_name not in fields_info:
                                    field_value = annot_obj.get("/V", "")
                                    if isinstance(field_value, bytes):
                                        field_value = field_value.decode('utf-8', errors='ignore')
                                    elif field_value:
                                        field_value = str(field_value)
                                    else:
                                        field_value = ""
                                    
                                    field_type = annot_obj.get("/FT", "Unknown")
                                    if isinstance(field_type, bytes):
                                        field_type = field_type.decode('utf-8', errors='ignore')
                                    else:
                                        field_type = str(field_type)
                                    
                                    fields_info[field_name] = {
                                        "type": field_type,
                                        "current_value": field_value,
                                        "page": page_num + 1
                                    }
                    except Exception as e:
                        continue
        except Exception as e:
            continue
    
    return fields_info


def fill_pdf_form(
    input_pdf_path: str,
    output_pdf_path: str,
    data: Dict[str, Any],
    preserve_appearance: bool = True
) -> bool:
    """
    Fill PDF form fields with provided data.
    Preserves logo and watermark by only modifying form field values.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_pdf_path: Path to save the filled PDF
        data: Dictionary mapping field names to values
        preserve_appearance: If True, tries to preserve visual appearance
    
    Returns:
        True if successful, False otherwise
    """
    if not PYPDF_AVAILABLE:
        print("Error: pypdf library is required. Install with: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(input_pdf_path)
        writer = PdfWriter()
        
        if reader.is_encrypted:
            if not reader.decrypt(""):
                print("Error: PDF is encrypted and requires a password.")
                return False
        
        # Copy all pages to preserve logo and watermark
        for page in reader.pages:
            writer.add_page(page)
        
        # Get root object
        root = reader.trailer.get("/Root", {})
        
        # Fill form fields from AcroForm
        if "/AcroForm" in root:
            try:
                acro_form = root["/AcroForm"]
                
                # Update the AcroForm in the writer
                if "/AcroForm" not in writer._root_object:
                    writer_acro_form = DictionaryObject()
                    writer._root_object[NameObject("/AcroForm")] = writer_acro_form
                else:
                    writer_acro_form = writer._root_object["/AcroForm"]
                
                # Copy AcroForm properties
                for key in acro_form:
                    if key != "/Fields":
                        try:
                            writer_acro_form[key] = acro_form[key]
                        except:
                            pass
                
                # Process fields
                if "/Fields" in acro_form:
                    fields = acro_form["/Fields"]
                    updated_fields = []
                    
                    for field_ref in fields:
                        try:
                            field_obj = field_ref.get_object() if hasattr(field_ref, 'get_object') else field_ref
                            
                            # Get field name
                            field_name_obj = field_obj.get("/T")
                            if field_name_obj:
                                if isinstance(field_name_obj, bytes):
                                    field_name = field_name_obj.decode('utf-8', errors='ignore')
                                else:
                                    field_name = str(field_name_obj)
                                
                                # Update field value if in data
                                if field_name in data:
                                    new_value = str(data[field_name])
                                    field_obj[NameObject("/V")] = TextStringObject(new_value)
                                    
                                    # Update appearance if needed
                                    if preserve_appearance:
                                        if "/AP" not in field_obj:
                                            field_obj[NameObject("/AP")] = DictionaryObject()
                                        field_obj[NameObject("/AP")][NameObject("/N")] = TextStringObject(new_value)
                            
                            updated_fields.append(field_ref)
                        except Exception as e:
                            print(f"Warning: Could not update field: {e}")
                            updated_fields.append(field_ref)
                    
                    writer_acro_form[NameObject("/Fields")] = updated_fields
            except Exception as e:
                print(f"Warning: Could not process AcroForm: {e}")
        
        # Also update annotations in pages (widget annotations)
        for page_num, page in enumerate(writer.pages):
            try:
                if "/Annots" in page:
                    annotations = page["/Annots"]
                    for annot_ref in annotations:
                        try:
                            annot_obj = annot_ref.get_object() if hasattr(annot_ref, 'get_object') else annot_ref
                            if "/Subtype" in annot_obj and annot_obj["/Subtype"] == "/Widget":
                                if "/T" in annot_obj:
                                    field_name_obj = annot_obj["/T"]
                                    if isinstance(field_name_obj, bytes):
                                        field_name = field_name_obj.decode('utf-8', errors='ignore')
                                    else:
                                        field_name = str(field_name_obj)
                                    
                                    if field_name in data:
                                        new_value = str(data[field_name])
                                        annot_obj[NameObject("/V")] = TextStringObject(new_value)
                                        
                                        # Update appearance
                                        if preserve_appearance:
                                            if "/AP" not in annot_obj:
                                                annot_obj[NameObject("/AP")] = DictionaryObject()
                                            annot_obj[NameObject("/AP")][NameObject("/N")] = TextStringObject(new_value)
                        except Exception as e:
                            continue
            except Exception as e:
                continue
        
        # Write output PDF
        with open(output_pdf_path, "wb") as output_file:
            writer.write(output_file)
        
        return True
        
    except Exception as e:
        print(f"Error filling PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to demonstrate PDF form filling.
    """
    pdf_path = Path("sample.pdf")
    
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found!")
        return
    
    print(f"Inspecting PDF: {pdf_path}")
    print("-" * 50)
    
    # Inspect PDF fields
    fields = inspect_pdf_fields(str(pdf_path))
    
    if not fields:
        print("No form fields found in PDF.")
        print("\nNote: If the PDF doesn't have form fields, you may need to:")
        print("1. Use a PDF editor to add form fields, or")
        print("2. Use a different approach like overlay text on specific coordinates")
        return
    
    print(f"Found {len(fields)} form field(s):\n")
    for field_name, field_info in fields.items():
        print(f"  Field: {field_name}")
        print(f"    Type: {field_info.get('type', 'Unknown')}")
        print(f"    Current Value: {field_info.get('current_value', 'Empty')}")
        if 'page' in field_info:
            print(f"    Page: {field_info['page']}")
        print()
    
    # Example data to fill
    sample_data = {
        # Add your field names and values here
        # Example: "Name": "John Doe",
        #          "Address": "123 Main St, City, State",
    }
    
    # Auto-fill with sample data if fields match common patterns
    for field_name in fields.keys():
        field_lower = field_name.lower()
        if "name" in field_lower and "name" not in sample_data:
            sample_data[field_name] = "John Doe"
        elif "address" in field_lower and "address" not in sample_data:
            sample_data[field_name] = "123 Main Street, Mumbai, Maharashtra 400001"
        elif "date" in field_lower and "date" not in sample_data:
            sample_data[field_name] = "2024-01-15"
        elif "pan" in field_lower and "pan" not in sample_data:
            sample_data[field_name] = "ABCDE1234F"
        elif "phone" in field_lower or "mobile" in field_lower:
            if "phone" not in sample_data and "mobile" not in sample_data:
                sample_data[field_name] = "9876543210"
    
    if sample_data:
        print("Filling PDF with sample data:")
        for key, value in sample_data.items():
            print(f"  {key}: {value}")
        print()
        
        output_path = Path("sample_filled.pdf")
        success = fill_pdf_form(
            str(pdf_path),
            str(output_path),
            sample_data,
            preserve_appearance=True
        )
        
        if success:
            print(f"\n✓ PDF filled successfully! Output: {output_path}")
        else:
            print("\n✗ Failed to fill PDF.")
    else:
        print("\nNo sample data to fill. Modify the script to add your data.")


if __name__ == "__main__":
    main()

