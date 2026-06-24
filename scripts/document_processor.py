import os
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from pdf2image import convert_from_path

def extract_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text().strip()
        
        # If the page has very little text, it might be a scanned image
        if len(page_text) < 50:
            print(f"Page {page_num + 1} seems to be a scanned image. Running OCR...")
            page_text = extract_using_ocr(file_path, page_num)
        
        text += page_text + "\n"
        
    # Clean text (replace PUA bullet points to avoid tokenizer mismatches)
    text = text.replace('\uf0b7', '•')
    return text

def extract_using_ocr(file_path, page_num):
    """
    Extracts text from a specific page of a PDF using OCR.
    Note: Requires Poppler and Tesseract to be installed on the system.
    """
    try:
        # Convert specific page to image (1-indexed for pdf2image)
        images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1)
        if images:
            image = images[0]
            # Run Tesseract OCR
            text = pytesseract.image_to_string(image)
            return text
        return ""
    except Exception as e:
        print(f"OCR failed on page {page_num+1}: {e}")
        return ""

def process_document(file_path):
    """
    Main entry point for processing any supported document type.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'txt':
        return extract_from_txt(file_path)
    elif ext == 'docx':
        return extract_from_docx(file_path)
    elif ext == 'pdf':
        return extract_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: txt, docx, pdf")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = process_document(sys.argv[1])
        print("--- EXTRACTED TEXT ---")
        print(text[:500] + "..." if len(text) > 500 else text)
