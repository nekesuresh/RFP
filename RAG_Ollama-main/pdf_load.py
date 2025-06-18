import PyPDF2
import nltk
nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF and returns a list of (page_num, paragraph) tuples.
    """
    paragraphs = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            # Split by double newlines or use NLTK sent_tokenize for fallback
            raw_paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
            if not raw_paragraphs:
                # fallback: treat the whole page as one paragraph
                raw_paragraphs = [page_text.strip()]
            for para in raw_paragraphs:
                if para:
                    paragraphs.append((page_num, para))
    return paragraphs

def split_pdf_into_chunks_with_metadata(paragraphs, chunk_size=500, overlap=50):
    """
    Given a list of (page_num, paragraph) tuples, split into chunks with metadata.
    Returns a list of dicts: {"text": ..., "page": ..., "para": ...}
    """
    from nltk.tokenize import sent_tokenize
    chunks = []
    for page_num, para in paragraphs:
        sentences = sent_tokenize(para)
        current_chunk = ""
        for idx, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip(), "page": page_num, "para": para[:40] + ("..." if len(para) > 40 else "")})
                # Add overlap from end of last chunk
                if overlap > 0 and len(chunks) > 0:
                    overlap_words = current_chunk.strip().split()[-overlap:]
                    overlap_text = " ".join(overlap_words)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
        if current_chunk:
            chunks.append({"text": current_chunk.strip(), "page": page_num, "para": para[:40] + ("..." if len(para) > 40 else "")})
    return chunks

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            chunks.append(current_chunk.strip())
            # Add overlap from end of last chunk
            if overlap > 0 and len(chunks) > 0:
                overlap_words = current_chunk.strip().split()[-overlap:]
                overlap_text = " ".join(overlap_words)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks