import os
import re
import faiss
import numpy as np
import pdfplumber  # You'll need to install this: pip install pdfplumber

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- Configuration ---
# Make sure your API_KEY is set as an environment variable or replace the string below.
API_KEY = "AIzaSyBK8mGnBUsRoQPVmZaITMG0KkfYkEmCUP4"
if GENAI_AVAILABLE and API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=API_KEY)
else:
    GENAI_AVAILABLE = False
    print("Warning: API_KEY is not set or google.generativeai is not installed. This script cannot run.")
    exit()

# Model for creating embeddings
EMBEDDING_MODEL = "models/text-embedding-004"

# --- File Paths ---
PDF_PATH = "The Encyclopedia O fNatural Medicine.pdf"
FAISS_INDEX_PATH = "natural_medicine.index"
TEXT_CHUNKS_PATH = "natural_medicine_chunks.txt"
EMBEDDING_DIMENSION = 768 # Dimension for text-embedding-004

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file, preserving paragraphs."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None
    print(f"Starting text extraction from '{pdf_path}'...")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            print(f"  - Processing page {i+1}/{total_pages}")
            full_text += page.extract_text() + "\n"
    print("Text extraction complete.")
    return full_text

def chunk_text(text, chunk_size=1200, overlap=100):
    """Splits text into overlapping chunks based on paragraphs."""
    print("Chunking text...")
    # Clean up excessive newlines and spaces
    text = re.sub(r'\s*\n\s*', '\n', text).strip()
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 10]

    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) + 1 < chunk_size:
            current_chunk += p + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p[-overlap:] + "\n" # Start next chunk with overlap
    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Created {len(chunks)} text chunks.")
    return chunks

def create_embeddings(chunks):
    """Creates embeddings for a list of text chunks using the Gemini API."""
    print("Creating embeddings for text chunks...")
    embeddings = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"  - Embedding chunk {i+1}/{total_chunks}")
        try:
            result = genai.embed_content(model=EMBEDDING_MODEL, content=chunk, task_type="RETRIEVAL_DOCUMENT")
            embeddings.append(result['embedding'])
        except Exception as e:
            print(f"Error creating embedding for chunk {i+1}: {e}")
            # Add a zero vector as a placeholder to avoid breaking the index
            embeddings.append([0.0] * EMBEDDING_DIMENSION)
            
    print("Embedding creation complete.")
    return np.array(embeddings).astype('float32')

def main():
    """Main function to process the PDF and create RAG files."""
    if not GENAI_AVAILABLE:
        return

    # 1. Extract text from the PDF
    book_text = extract_text_from_pdf(PDF_PATH)
    if not book_text:
        return

    # 2. Chunk the text
    text_chunks = chunk_text(book_text)

    # 3. Create embeddings for each chunk
    embeddings = create_embeddings(text_chunks)

    # 4. Create and save the FAISS index
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}'")

    # 5. Save the text chunks
    print("Saving text chunks...")
    with open(TEXT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(text_chunks))
    print(f"Text chunks saved to '{TEXT_CHUNKS_PATH}'")

    print("\nProcessing complete! Your chatbot is ready to be updated.")

if __name__ == "__main__":
    main()
