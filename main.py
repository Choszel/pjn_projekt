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

if __name__ == '__main__':
    main()
