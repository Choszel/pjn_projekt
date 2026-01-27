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
        self.master.geometry("1050x525") 
        try:
            with open("retrieverBM25.pkl", "rb") as f:
                self.retriever = pickle.load(f)
        except FileNotFoundError:
            messagebox.showwarning("showwarning", f"Model dla domyślnego algorytmu (BM25) nie został znaleziony. Proszę przetworzyć dane.")
        
        try:
            with open("extractor.pkl", "rb") as f:
                self.extractor = pickle.load(f)
        except FileNotFoundError:
            messagebox.showwarning("showwarning", f"Model ekstraktora nie został znaleziony. Proszę przetworzyć dane.")

        self.create_widgets()

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

    def algoithm_changed(self, selected_algorithm):
        print(f"Wybrano algorytm: {selected_algorithm}")
        try:
            with open("retriever" + selected_algorithm + ".pkl", "rb") as f:
                self.retriever = pickle.load(f)
            messagebox.showinfo("showinfo", f"Załadowano model dla algorytmu {selected_algorithm}.") 
        except FileNotFoundError:
            messagebox.showwarning("showwarning", f"Model dla algorytmu {selected_algorithm} nie został znaleziony. Proszę przetworzyć dane. Dalej używany jest poprzedni model.")


    def create_widgets(self):
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill = tk.BOTH, expand = True)

        self.upper_frame = tk.Frame(self.main_frame)
        self.upper_frame.pack(fill = tk.X, padx = 10, pady = 10)

        tk.Label(self.upper_frame, text="O co chcesz zapytać?").pack(side = tk.LEFT)
        self.question_entry = tk.Entry(self.upper_frame, width = 85)
        self.question_entry.pack(side = tk.LEFT, padx = 5)

        self.submit_question_button = tk.Button(self.upper_frame, text="Zapytaj", command=self.ask_question)
        self.submit_question_button.pack(side = tk.LEFT, padx = 5)

        algorithms = ["BM25", "SentenceBERT", "TFIDF"]  
        self.chosen_algorithm = tk.StringVar(value="BM25")  
        self.algorithm_menu = tk.OptionMenu(self.upper_frame, self.chosen_algorithm, *algorithms, command = self.algoithm_changed).pack(side= tk.LEFT, padx = 5) 

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

            self.retriever = Retriever(data, nlp_model=processor.nlp, algorithm=self.chosen_algorithm.get())
            self.extractor = AnswerExtractor(nlp_model=processor.nlp)

            with open("retriever" + self.chosen_algorithm.get() + ".pkl", "wb") as f:
                pickle.dump(self.retriever, f)
            with open("extractor.pkl", "wb") as f:
                pickle.dump(self.extractor, f)

        except Exception as X:
            messagebox.showerror("showerror", X)
            return

        messagebox.showinfo("showinfo", "Załadowano model i przetworzono dane.") 

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
