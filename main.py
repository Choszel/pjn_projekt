import os
import tkinter as tk
from data_processor import DataProcessor
from retriever import Retriever
from answer_extractor import AnswerExtractor
import pickle
from tkinter import messagebox 

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("System Pytań i Odpowiedzi")
        with open("retriever.pkl", "rb") as f:
            self.retriever = pickle.load(f)

        with open("extractor.pkl", "rb") as f:
            self.extractor = pickle.load(f)

        self.master.geometry("1000x500")  # Set window size
        self.create_widgets()

        # print("\n--- 3. Eksperymenty i Ocena  ---")

    def ask_question(self):
        question = self.question_entry.get()
        print(f"Pytanie: {question}")

        self.answers_text.configure(state = tk.NORMAL)
        self.answers_text.insert(tk.END, f"{question}\n", "question")
        results = self.retriever.retrieve_top_k(question, k = 1)

        if results:
            top_result, score = results[0]
            answer = self.extractor.get_best_answer(question, top_result)
            print(answer, score)
            self.answers_text.insert(tk.END, f"{answer}\n\n", "answer")
        else:
            print("Nie znaleziono odpowiedzi.")
            self.answers_text.insert(tk.END, "Nie znaleziono odpowiedzi.\n\n", "answer")

        self.answers_text.see(tk.END)
        self.answers_text.configure(state = tk.DISABLED)
        self.question_entry.delete(0, tk.END)

    def create_widgets(self):
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill = tk.BOTH, expand = True)

        self.upper_frame = tk.Frame(self.main_frame)
        self.upper_frame.pack(fill = tk.X, padx = 10, pady = 10)

        tk.Label(self.upper_frame, text="O co chcesz zapytać?").pack(side = tk.LEFT)
        self.question_entry = tk.Entry(self.upper_frame, width = 100)
        self.question_entry.pack(side = tk.LEFT, padx = 5)

        self.submit_question_button = tk.Button(self.upper_frame, text="Zapytaj", command=self.ask_question)
        self.submit_question_button.pack(side = tk.LEFT, padx = 5)

        self.prepare_data_button = tk.Button(self.upper_frame, text="Przetwórz dane i załaduj model", command=self.prepare_data)
        self.prepare_data_button.pack(side = tk.LEFT, padx = 5)

        self.answers_frame = tk.Frame(self.main_frame)
        self.answers_frame.pack(side = tk.TOP, fill = tk.BOTH, expand=True, padx = 10, pady = 10)

        self.answers_scroll_bar = tk.Scrollbar(self.answers_frame)
        self.answers_scroll_bar.pack(side = tk.RIGHT, fill = tk.Y)

        self.answers_text = tk.Text(self.answers_frame, wrap = tk.WORD, yscrollcommand = self.answers_scroll_bar.set, state = tk.DISABLED)
        self.answers_text.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        self.answers_scroll_bar.config(command = self.answers_text.yview)

        self.answers_text.tag_configure("question", font = ("Arial", 11, "bold"))
        self.answers_text.tag_configure("answer", font = ("Arial", 11))


    def prepare_data(self):
        print("--- 1. Przygotowanie Danych ---")

        directory= "documents"
        processor = DataProcessor()
        data = []

        try:
            for filename in os.listdir(directory):
                input_path = os.path.join(directory, filename)
                file_data = processor.process_document(input_path)

                for item in file_data:
                    item["source"] = filename

                data.extend(file_data)

            self.retriever = Retriever(data, nlp_model=processor.nlp)
            self.extractor = AnswerExtractor(nlp_model=processor.nlp)

            with open("retriever.pkl", "wb") as f:
                pickle.dump(self.retriever, f)
            with open("extractor.pkl", "wb") as f:
                pickle.dump(self.extractor, f)

        except Exception as X:
            messagebox.showerror("showerror", X)

        messagebox.showinfo("showinfo", "Załadowano model i przetworzono dane.") 

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
