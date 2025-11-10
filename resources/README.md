# Resources Directory

Place your custom branding files here to customize the PDF reports.

## Logo
- **File:** `logo.png`, `logo.jpg`, or `logo.jpeg`
- **Recommended size:** 200x60 pixels (or similar aspect ratio)
- **Format:** PNG, JPG, or JPEG
- **Location:** Place directly in this `resources/` directory
- The logo will appear at the top-left of every PDF report
- If not provided, a text-based placeholder will be used

## Footer Image (Optional)
- **File:** `footer.png`, `footer.jpg`, or `footer.jpeg`
- **Recommended size:** Full page width (8.5 inches), 100-150 pixels height
- **Format:** PNG, JPG, or JPEG
- **Location:** Place directly in this `resources/` directory
- If not provided, a colored bar footer will be generated automatically using colors from `config.json`

## Footer Configuration
- **File:** `config.json` (already created)
- Edit this file to customize:
  - Address
  - Website URL
  - Email
  - Phone number
  - Footer bar colors (array of hex colors)

## Example Usage

1. **Add your logo:**
   ```
   cp /path/to/your/logo.png resources/logo.png
   ```

2. **Add footer image (optional):**
   ```
   cp /path/to/your/footer.png resources/footer.png
   ```

3. **Customize footer text:**
   Edit `resources/config.json` with your contact information

## Notes
- Supported image formats: PNG, JPG, JPEG
- Logo aspect ratio will be maintained automatically
- Footer image will be scaled to fit page width
- All files should be placed directly in this `resources/` directory
- Changes take effect immediately (no server restart needed for new files)

