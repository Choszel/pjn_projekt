import os
from data_processor import DataProcessor
from retriever import Retriever
from answer_extractor import AnswerExtractor

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
    extractor = AnswerExtractor(nlp_model=processor.nlp)

    question = ("Ile kosztuje wydanie elektronicznej legitymacji studenckiej?")

    while True:
        user_question = input("o co chcesz zapytać?\n")

        if user_question == "q":
            break

        print(f"Pytanie: {user_question}")

        results = retriever.retrieve_top_k(user_question, k=1)

        if results:
            top_result, score = results[0]
            answer = extractor.get_best_answer(user_question,top_result)

            print(answer)
        else:
            print("nie znaleziono odpowiedzi.")

     # print("\n--- 3. Eksperymenty i Ocena  ---")


if __name__ == '__main__':
    main()
