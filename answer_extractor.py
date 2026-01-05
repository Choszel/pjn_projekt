import re
from typing import Optional, Dict

class AnswerExtractor:

    def __init__(self, nlp_model):

        self.nlp = nlp_model

    def extract_answer_classic_nlp(self, question: str, context_dict: Dict) -> Optional[str]:

        text = context_dict.get("content", "")
        doc_context = self.nlp(text)
        question_lower = question.lower()

        #  Logika dla kwot i liczb (np. "Ile kosztuje...", "Jaka jest opłata...")
        money_keywords = ["ile", "wynosi", "kwota", "liczba", "wysokość", "koszt", "cena", "opłata", "złotych", "zł"]
        if any(w in question_lower for w in money_keywords):
            for ent in doc_context.ents:
                if ent.label_ in ("amount", "money", "quantity", "val"):
                    return ent.text.strip()

            regex_money = re.search(r'\d+(?:[\s,.]\d+)?\s*(?:zł|pln|złotych)', text, re.IGNORECASE)
            if regex_money:
                return regex_money.group()

        #  Logika dla dat (np. "Kiedy...", "Od którego...")
        date_keywords = ["termin", "do", "od", "kiedy", "deadline", "data"]
        if any(k in q for k in date_keywords):
            for ent in doc_context.ents:
                if ent.label_ in ("date", "time"):
                    return ent.text.strip()

        # miejsce
        place_keywords = ["miejsce", "w", "na", "adres", "lokalizacja"]

        if any(k in q for k in place_keywords):
            for ent in doc_context.ents:
                if ent.label_ in ("GPE", "LOC"):
                    return ent.text.strip()

        #osoby i organizacje
        person_keywords = ["kto", "osoba", "organ", "organizacja", "instytucja"]

        if any(k in q for k in person_keywords):
            for ent in doc_context.ents:
                if ent.label_ in ("PERSON", "ORG"):
                    return ent.text.strip()

        return None

    def get_best_answer(self, question: str, context_dict: Dict) -> str:

        answer = self.extract_answer_classic_nlp(question, context_dict)

        if answer:
            return f"**Znaleziona informacja:** {answer}"

        source_info = f" [Źródło: {context_dict.get('source', 'nieznane')}]" if 'source' in context_dict else ""
        return f"**Fragment regulaminu:** {context_dict['content']}{source_info}"
