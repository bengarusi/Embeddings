# Document Indexing with Gemini Embeddings

A Python-based command-line tool for indexing textual documents (PDF and DOCX),
splitting them into meaningful text chunks, generating vector embeddings using
Google Gemini, and storing the results in a PostgreSQL database with pgvector support.

The project was implemented according to the assignment requirements and supports
running the indexing process using **a single chunking strategy per execution**.

---

## Features

- Supports PDF and DOCX documents
- Extracts and cleans textual content from documents
- Multiple chunking strategies (fixed, sentence-based, paragraph-based)
- Overlapping chunks to preserve semantic context
- Embedding generation using Google Gemini
- Persistent storage using PostgreSQL and pgvector
- Simple command-line interface

---

## Requirements

- Python 3.9+
- PostgreSQL with pgvector extension enabled

Python dependencies:
- psycopg
- python-dotenv
- pypdf
- python-docx
- google-generativeai (Gemini client)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY_NAME>.git
cd <REPOSITORY_NAME>
```
### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows
```

### 3. Install Python dependencies
```bash
pip install psycopg python-dotenv pypdf python-docx google-generativeai
```

## Database Setup
1. Install PostgreSQL
2. Enable the pgvector extension:
```bash
CREATE EXTENSION IF NOT EXISTS vector;
```
3. Create the document_chunks table:
```bash
CREATE TABLE document_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(3072) NOT NULL,
    filename TEXT NOT NULL,
    strategy_split TEXT NOT NULL,
    at_created TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

## Usage
The program is executed from the command line and requires:
1. a document file path
2. a chunking strategy

# Supported strategies
1.fixed    
2.sentence    
3.paragraph    

## Example commands
```bash
python index_documents.py --file data/test.pdf --strategy fixed
```

## How it Works
**1. Document Loading**
- PDF files are processed using pypdf
- DOCX files are processed using python-docx

**2. Text Cleaning**
- The extracted text is normalized to:
- remove unnecessary whitespace
- fix hyphenated line breaks
- preserve paragraph structure

**3. Chunking**
The cleaned text is split into chunks using the selected strategy.
Overlapping chunks are used to preserve contextual information.

**4. Embedding Generation**
Each chunk is converted into a vector embedding using the
gemini-embedding-001 model.

**5. Storage**
Chunks and their embeddings are stored in PostgreSQL using the pgvector type.


## Example Output

Running the program with a sample PDF document:

```bash
python index_documents.py --file data/test.pdf --strategy fixed

Created 6 chunks using strategy 'fixed'
Indexing completed successfully.
```

## Data Directory 
The data/ directory contains small sample documents used for testing and demonstration.

## Customization 
- Chunk sizes and overlap values can be adjusted in the configuration section of
index_documents.py

- Additional chunking strategies can be added by extending the chunking module


- The embedding model can be replaced by changing the model configuration



