import PyPDF2
import nltk
import re
from typing import List, Tuple, Dict, Any
import tiktoken

# Always download 'punkt' for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download required NLTK data
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

def get_tokenizer():
    """Get tiktoken tokenizer for accurate token counting"""
    try:
        return tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
    except:
        try:
            return tiktoken.get_encoding("gpt2")  # Fallback
        except:
            return None  # Will use word-based approximation

def clean_text(text: str) -> str:
    """Clean and normalize text for better tokenization"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix camelCase
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix sentence boundaries
    # Remove special characters that might interfere with tokenization
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', ' ', text)
    return text.strip()

def tokenize_sentences(text: str) -> List[str]:
    """Advanced sentence tokenization with technical content handling"""
    # Clean the text first
    text = clean_text(text)
    try:
        return nltk.sent_tokenize(text)
    except Exception as e:
        import logging
        logging.error(f"NLTK sentence tokenization failed: {e}. Falling back to simple split.")
        return text.split('.')

def tokenize_words(text: str) -> List[str]:
    """Word tokenization with technical term preservation"""
    # Use NLTK word tokenizer
    words = nltk.word_tokenize(text)
    
    # Preserve technical terms and acronyms
    refined_words = []
    for word in words:
        # Handle acronyms and technical terms
        if re.match(r'^[A-Z]{2,}$', word):  # All caps acronyms
            refined_words.append(word)
        elif re.match(r'^[a-zA-Z0-9\-_\.]+$', word):  # Technical terms with special chars
            refined_words.append(word)
        else:
            refined_words.append(word)
    
    return refined_words

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken or fallback to word-based approximation"""
    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except:
            pass
    
    # Fallback: approximate tokens as words * 1.3 (typical ratio)
    words = text.split()
    return int(len(words) * 1.3)

def split_by_tokens(text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    """Split text by token count rather than character count"""
    tokenizer = get_tokenizer()
    sentences = tokenize_sentences(text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += (" " if current_chunk else "") + sentence
            current_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Handle overlap
            if overlap_tokens > 0 and chunks:
                # Get last few sentences for overlap
                overlap_text = ""
                overlap_count = 0
                for sent in reversed(current_chunk.split('. ')):
                    sent_tokens = count_tokens(sent)
                    if overlap_count + sent_tokens <= overlap_tokens:
                        overlap_text = sent + (". " if overlap_text else "") + overlap_text
                        overlap_count += sent_tokens
                    else:
                        break
                current_chunk = overlap_text + " " + sentence
                current_tokens = count_tokens(current_chunk)
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from a PDF and returns a list of (page_num, paragraph) tuples.
    Uses improved text cleaning and paragraph detection.
    """
    paragraphs = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            
            # Clean the page text
            page_text = clean_text(page_text)
            
            # Split by paragraph markers (double newlines, section breaks, etc.)
            paragraph_markers = [
                r'\n\s*\n',           # Double newlines
                r'\n\s*[A-Z][A-Z\s]+\n',  # Section headers
                r'\n\s*\d+\.\s*\n',   # Numbered sections
                r'\n\s*[•\-]\s*\n',   # Bullet points
            ]
            
            raw_paragraphs = re.split('|'.join(paragraph_markers), page_text)
            raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
            
            if not raw_paragraphs:
                # Fallback: treat the whole page as one paragraph
                raw_paragraphs = [page_text.strip()]
            
            for para in raw_paragraphs:
                if para and len(para) > 10:  # Filter out very short paragraphs
                    paragraphs.append((page_num, para))
    
    return paragraphs

def split_pdf_into_chunks_with_metadata(paragraphs: List[Tuple[int, str]], 
                                       max_tokens: int = 500, 
                                       overlap_tokens: int = 50) -> List[Dict[str, Any]]:
    """
    Split PDF paragraphs into token-based chunks with metadata.
    Returns a list of dicts: {"text": ..., "page": ..., "para": ..., "tokens": ...}
    """
    chunks = []
    
    for page_num, para in paragraphs:
        # Split paragraph by tokens
        para_chunks = split_by_tokens(para, max_tokens, overlap_tokens)
        
        for chunk in para_chunks:
            if chunk.strip():
                chunk_info = {
                    "text": chunk.strip(),
                    "page": page_num,
                    "para": para[:60] + ("..." if len(para) > 60 else ""),
                    "tokens": count_tokens(chunk.strip()),
                    "characters": len(chunk.strip())
                }
                chunks.append(chunk_info)
    
    return chunks

def split_text_into_chunks(text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    """Split text into token-based chunks"""
    return split_by_tokens(text, max_tokens, overlap_tokens)

def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about the chunks for analysis"""
    if not chunks:
        return {}
    
    token_counts = [chunk.get('tokens', 0) for chunk in chunks]
    char_counts = [chunk.get('characters', 0) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "total_characters": sum(char_counts),
        "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
        "avg_chars_per_chunk": sum(char_counts) / len(chunks),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts)
    }