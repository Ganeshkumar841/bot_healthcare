import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import google.generativeai as genai

# --- Configuration ---
# IMPORTANT: Use the same API key as your main app.
API_KEY = os.getenv("API_KEY", "AIzaSyBK8mGnBUsRoQPVmZaITMG0KkfYkEmCUP4")
genai.configure(api_key=API_KEY)

# --- IMPORTANT ---
# This line has been updated to use your uploaded encyclopedia.
PDF_FILE_PATH = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition.pdf" 
FAISS_INDEX_PATH = "health_book.index"
TEXT_CHUNKS_PATH = "health_book_chunks.txt"

# This is the model you'll use for creating embeddings.
EMBEDDING_MODEL = "models/text-embedding-004"

# --- Functions ---

def extract_text_from_pdf(pdf_path):
    """Opens a PDF and extracts its text content."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} was not found.")
    print(f"Reading text from {pdf_path}...")
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"An error occurred while trying to read the PDF: {e}")
        print("This often means the PDF file is corrupted or in an incompatible format.")
        return None

    print("Text extraction complete.")
    return text

def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    """Splits a long text into smaller, overlapping chunks."""
    print("Splitting text into chunks...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"Created {len(chunks)} chunks.")
    return chunks

def save_chunks_to_file(chunks, file_path):
    """Saves the text chunks to a file for later retrieval."""
    print(f"Saving text chunks to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n---\n")
    print("Chunks saved.")

def create_faiss_index(chunks):
    """Creates a FAISS index from text chunks using Gemini embeddings."""
    print("Generating embeddings for chunks. This may take a while...")
    try:
        # Generate embeddings for all chunks. The API handles batching automatically.
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = result['embedding']
        
        # Ensure all embeddings are NumPy arrays of the same float type
        embeddings = np.array(embeddings).astype('float32')

        # Create a FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        print(f"FAISS index created with {index.ntotal} vectors.")
        return index

    except Exception as e:
        print(f"An error occurred during embedding generation or FAISS indexing: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Extract text from the PDF
        book_text = extract_text_from_pdf(PDF_FILE_PATH)
        
        # Check if text extraction was successful
        if not book_text or not book_text.strip():
            print("\nSetup failed: No text could be extracted from the PDF.")
            print("The PDF file may be corrupted, image-based, or password-protected.")
            exit() # Stop the script here

        # 2. Split the text into chunks
        text_chunks = split_text_into_chunks(book_text)
        
        if not text_chunks:
            print("\nSetup failed: Text was extracted, but no chunks could be created.")
            exit()

        # 3. Save chunks for reference (optional but good for debugging)
        save_chunks_to_file(text_chunks, TEXT_CHUNKS_PATH)
        
        # 4. Create and save the FAISS index
        faiss_index = create_faiss_index(text_chunks)
        if faiss_index:
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            print(f"FAISS index successfully saved to {FAISS_INDEX_PATH}")
            print("\nSetup complete! You can now run the main app.py.")
        else:
            print("\nSetup failed. Please check the error messages above.")

    except FileNotFoundError as e:
        print(e)
        print("Please make sure your PDF file is in the same directory and the PDF_FILE_PATH is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

