import re
import fitz
import spacy

class DataProcessor:

    def __init__(self):
        try:
            self.nlp = spacy.load("pl_core_news_sm")
        except OSError:
            self.nlp = None

    def load_pdf(self, file_path):

        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text

    def clean_text(self, text):

        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\.\s*\.\s*\.\s*', '', text)

        return text.strip()

    def get_chunks(self, text):

        pattern = r'(\n\s*(?:§\s*\d+|(?:\d+\.)|(?:\d+\)))\s+)'

        parts = re.split(pattern, text)

        refined_chunks = []
        if parts[0].strip():
            refined_chunks.append(parts[0].strip())

        for i in range(1, len(parts), 2):
            marker = parts[i]
            content = parts[i+1] if i+1 < len(parts) else ""
            full_chunk = (marker + content).strip()

            if full_chunk:
                full_chunk = re.sub(r'\n+', ' ', full_chunk)
                refined_chunks.append(full_chunk)

        final_passages = []
        buffer = ""

        for chunk in refined_chunks:
            if buffer:
                chunk = buffer + "\n" + chunk
                buffer = ""

            if len(chunk) < 50 and not chunk.endswith(('.', ';', ':')):
                buffer = chunk
            else:
                final_passages.append(chunk)

        if buffer:
            final_passages.append(buffer)

        return final_passages

    def preprocess_for_search(self, text):

        if not self.nlp:
            return text.lower()

        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc
                  if not token.is_punct and not token.is_space and not token.is_stop]
        return " ".join(tokens)

    def process_document(self, file_path):

        raw_text = self.load_pdf(file_path)
        cleaned_text = self.clean_text(raw_text)
        passages = self.get_chunks(cleaned_text)

        processed_data = []
        for i, p in enumerate(passages):
            processed_data.append({
                "id": i,
                "content": p,
                "lemmatized": self.preprocess_for_search(p)
            })

        return processed_data
