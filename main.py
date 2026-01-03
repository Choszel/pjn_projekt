import os
from data_processor import DataProcessor
from retriever import Retriever

def main():
    print("--- 1. Przygotowanie Danych ---")

    directory= "documents"
    processor = DataProcessor()
    data = []

    for filename in os.listdir(directory):
        input_path = os.path.join(directory, filename)
        file_data = processor.process_document(input_path)

        for item in file_data:
            item["source"] = filename

        data.extend(file_data)

    print("\n--- 2. Działanie ---")

    retriever = Retriever(data, nlp_model=processor.nlp)

    question = "Ile kosztuje wydanie elektronicznej legitymacji studenckiej?"
    print(f"Zapytanie: {question}\n")

    results = retriever.retrieve_top_k(question, k=3)

    if not results:
        print("Nie znaleziono pasujących fragmentów.")
    else:
        for i, (pasus, score) in enumerate(results, 1):
            print(f"Wynik #{i} (Score: {score:.2f}) | Źródło: {pasus.get('source', 'Nieznane')}")
            print(f"Treść: {pasus['content']}")
            print("-" * 50)

if __name__ == '__main__':
    main()
