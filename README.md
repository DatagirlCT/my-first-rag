# my-first-rag

## Overview

This Python script extracts text from specified URLs, processes it, and uses OpenAI's language model to answer a given question based on the extracted content.

## Features

- **Extracts text from web pages**: Fetches and extracts plain text from a list of URLs.
- **Processes PDF files**: (Note: PDF functionality is included but not utilized in the current script.)
- **Text Chunking**: Splits text into chunks for better handling by the language model.
- **Embedding and Vector Store**: Uses OpenAI embeddings to store and search text chunks.
- **Question Answering**: Uses a language model to answer a predefined question based on the extracted text.

## Requirements

- Python 3.x
- `requests`
- `beautifulsoup4`
- `langchain_openai`
- `langchain_community`
- `pymupdf` (PyMuPDF)
- `chromadb`

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Add your OpenAI API key

Go here to create and get your key: https://platform.openai.com/api-keys

You can add the api key to your environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
```

