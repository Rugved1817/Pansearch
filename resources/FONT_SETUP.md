# Devanagari Font Setup for PDF Generation

This document explains how to set up Devanagari (Marathi/Hindi) fonts for proper rendering in generated PDFs.

## Problem

If Marathi/Devanagari text appears as garbled characters or boxes in the PDF, it means a Devanagari-supporting font is not available or not properly registered.

## Solution

### Option 1: Bundle Font in Resources (Recommended)

1. Download Noto Sans Devanagari font from Google Fonts:
   - Visit: https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
   - Click "Download family" to get the ZIP file
   - Extract the ZIP file
   - Copy `NotoSansDevanagari-Regular.ttf` to the `resources` folder
   - (Optional) Copy `NotoSansDevanagari-Bold.ttf` for bold text support

2. The application will automatically detect and use fonts in the `resources` folder.

### Option 2: Install System-Wide (Windows)

1. Download Noto Sans Devanagari font (see Option 1)
2. Copy `NotoSansDevanagari-Regular.ttf` to `C:\Windows\Fonts\`
3. The application will automatically detect system-installed fonts

### Option 3: Enable Hindi Language Support (Windows)

1. Go to Windows Settings > Time & Language > Language
2. Add Hindi language support
3. This will install the Mangal font, which supports Devanagari
4. The application will automatically detect the Mangal font

### Option 4: Install on Linux

```bash
sudo apt-get install fonts-noto-core
```

### Option 5: Install on macOS

Noto Sans Devanagari is usually pre-installed on macOS. If not:
1. Download from Google Fonts (see Option 1)
2. Install by double-clicking the font file
3. Or copy to `/Library/Fonts/` or `~/Library/Fonts/`

## Verification

After installing the font, restart the application and check the logs. You should see:
```
INFO: Registered Devanagari regular font: /path/to/font
INFO: Devanagari font support enabled: regular=Devanagari, bold=Devanagari-Bold
```

If you see warnings about fonts not being found, the font is not properly installed.

## Font Locations Checked

The application checks for fonts in this order:

1. `resources/NotoSansDevanagari-Regular.ttf` (bundled font)
2. `resources/fonts/NotoSansDevanagari-Regular.ttf` (bundled font in subfolder)
3. System font directories (Windows, macOS, Linux)
4. Windows: `C:\Windows\Fonts\Mangal.ttf` (if Hindi language support is installed)

## Troubleshooting

### Font not found
- Verify the font file exists in one of the checked locations
- Check file permissions (application must be able to read the font file)
- Restart the application after installing fonts

### Font found but text still garbled
- Verify the font file is not corrupted (try opening it in a font viewer)
- Check application logs for font registration errors
- Ensure the text is properly encoded as UTF-8

### Partial rendering issues
- Some complex Devanagari characters may not render correctly in older fonts
- Use Noto Sans Devanagari (latest version) for best compatibility
- Check that both regular and bold fonts are installed if using bold text

## Notes

- ReportLab automatically embeds TTF fonts in the PDF, so recipients don't need the font installed
- Unicode normalization (NFC) is applied to ensure proper character rendering
- The application falls back to Helvetica if no Devanagari font is found (text will be garbled)

