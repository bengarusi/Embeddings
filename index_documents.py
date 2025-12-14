import argparse # Command-line argument parsing
import os
import re # Regular expressions for text cleaning
from pathlib import Path
from typing import List
import psycopg # PostgreSQL database connection
from dotenv import load_dotenv # Load environment variables from .env file
from pypdf import PdfReader
from docx import Document
from google import genai # Gemini API client


# -------------------- Config --------------------
EMBEDDING_MODEL = "gemini-embedding-001"

# Supported file types for indexing
SUPPORTED_EXTENSIONS = {".pdf", ".docx"} 

# Fixed chunk strategy parameters
FIXED_CHUNK_SIZE = 1000
FIXED_OVERLAP = 200

# Paragraph chunk strategy parameters
PARAGRAPH_MAX_CHARS = 1800
PARAGRAPH_OVERLAP = 1  

# Sentence chunk strategy parameters
SENTENCE_MAX_CHARS = 1200
SENTENCE_OVERLAP = 1  



# -------------------- Utils --------------------
def clean_text(text: str) -> str:
    """
    Normalize and clean extracted text to improve chunking quality:
    - Normalize newlines
    - Fix hyphenated line-breaks (word-\nword -> wordword)
    - Collapse multiple spaces/tabs
    - Normalize paragraph breaks into '\n\n'
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n") # normalize newlines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # fix hyphen line breaks
    text = re.sub(r"[ \t]+", " ", text) # collapse multiple spaces/tabs to single space for noise reduction
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip() 


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    # Try to extract text from each page (fallback to empty string if extraction fails)
    return "\n".join(page.extract_text() or "" for page in reader.pages) 


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    # Extract text from each paragraph (skip empty paragraphs)
    return "\n".join(p.text for p in doc.paragraphs if p.text)


# -------------------- Chunking --------------------

# Fixed-size chunking with overlap
def chunk_fixed(text: str, size: int, overlap: int) -> List[str]:
    """
    Fixed-size chunking with character overlap.
    Example: size=1000, overlap=200 means:
    chunk1: [0..1000)
    chunk2: [800..1800)
    ...
    """

    # Validate parameters to avoid infinite loops or invalid slicing
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

        
    chunks = []
    start = 0
    # Create chunks until we reach end of text
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        # Avoid inserting empty chunks
        if chunk:
            chunks.append(chunk)
        # Move start forward, but keep overlap from previous chunk
        start = end - overlap 
    return chunks


def chunk_units_by_char_limit(units: List[str], max_chars: int, overlap_units: int, sep: str) -> List[str]:
    """
    Generic chunk builder:
    - units: list of sentences/paragraphs
    - max_chars: max characters per chunk (soft limit)
    - overlap_units: how many units to overlap between chunks (for context continuity)
    - sep: separator to join units inside a chunk

    Note: We allow at least one unit even if it exceeds max_chars (to avoid infinite loop).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_units < 0:
        raise ValueError("overlap_units must be >= 0")

    chunks: List[str] = []
    start = 0

    # Build chunks until all units are consumed
    while start < len(units):
        buf = ""
        end = start

        while end < len(units):
            candidate = (buf + sep + units[end]).strip() if buf else units[end]
            # Allow at least one unit even if it exceeds max_chars
            if len(candidate) <= max_chars or not buf:
                buf = candidate
                end += 1
            else:
                break

        if buf:
            chunks.append(buf)

        if end >= len(units):
            break

        start = max(end - overlap_units, start + 1)

    return chunks



# Sentence-based chunking
def chunk_sentence(text: str, max_chars: int, overlap_sentences: int = 0) -> List[str]:
    """
    Sentence-based chunking:
    - Split text into sentences using punctuation (. ! ?)
    - Group sentences into chunks limited by max_chars
    - Overlap a configurable number of sentences
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return chunk_units_by_char_limit(sentences, max_chars, overlap_sentences, sep=" ")

# Paragraph-based chunking
def chunk_paragraph(text: str, max_chars: int, overlap_paragraphs: int=1) -> List[str]:
    """
    Paragraph-based chunking:
    - Split text into paragraphs using blank lines (\n\n)
    - Group paragraphs into chunks limited by max_chars
    - Overlap a configurable number of paragraphs
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return chunk_units_by_char_limit(paragraphs, max_chars, overlap_paragraphs, sep="\n\n")




# -------------------- Main --------------------
def main():
    # Load environment variables from .env into os.environ
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--strategy", required=True, choices=["fixed", "sentence", "paragraph"])
 
    args = parser.parse_args()

    # Validate file extension
    file_path = Path(args.file)
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type")

    # Read file
    if file_path.suffix.lower() == ".pdf":
        text = read_pdf(file_path)
    else:
        text = read_docx(file_path)

    # Clean text
    text = clean_text(text)

    if not text.strip():
            raise ValueError(f"No extractable text found in file: {file_path.name}")


    # Chunking 
    if args.strategy == "fixed":
        chunks = chunk_fixed(text, FIXED_CHUNK_SIZE, FIXED_OVERLAP)
    elif args.strategy == "sentence":
        chunks = chunk_sentence(text, SENTENCE_MAX_CHARS, SENTENCE_OVERLAP)
    else:
        chunks = chunk_paragraph(text, PARAGRAPH_MAX_CHARS , PARAGRAPH_OVERLAP)

    print(f"Created {len(chunks)} chunks using strategy '{args.strategy}'")

    # Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # DB connection
    conn = psycopg.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()


    # For each chunk: create embedding and store in DB
    for chunk in chunks:
        # Generate embedding using Gemini embedding endpoint
        emb = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk
        )
        vector = emb.embeddings[0].values

        cur.execute(
            """
            INSERT INTO document_chunks
            (chunk_text, embedding, filename, strategy_split)
            VALUES (%s, %s::vector, %s, %s)
            """,
            (
                chunk,
                str(vector),
                file_path.name,
                args.strategy
            )
        )

    conn.commit()
    cur.close()
    conn.close()
    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()
