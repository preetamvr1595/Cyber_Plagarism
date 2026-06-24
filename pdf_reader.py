import sys
import fitz

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeErrors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

pdf_path = input("Enter PDF path: ").strip().strip('"').strip("'")

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

# Replace non-standard PUA bullet points (\uf0b7) with standard Unicode bullets (•)
# to prevent console layout and cursor positioning bugs on Windows terminals
text = text.replace('\uf0b7', '•')

print(text[:5000])