import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import tkinter as tk
import nltk
import threading

nltk.download('punkt')

tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(entry, text, loading_label, input_text):
    try:
        loading_label.config(text="Loading...")

        sentences = nltk.sent_tokenize(input_text)

        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            inputs = {key: value.to(device) for key, value in inputs.items()}

            total_chars = sum(len(sentence) for sentence in sentences) if sentences else 0

            outputs = model.generate(**inputs, max_length=512, num_return_sequences=1, no_repeat_ngram_size=2)
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            type_out_response(text, input_text, " ".join(responses))

            text.after(0, lambda: update_text_widget(text, input_text, " ".join(responses)))

            entry.delete(0, tk.END)
    except Exception as e:
        response = f"Error: {str(e)}"
        text.after(0, lambda: update_text_widget(text, input_text, response))
    finally:
        loading_label.config(text="")

def type_out_response(text, user_input, response):
    def type_character(i):
        if i <= len(response):
            partial_response = response[:i]
            text.config(state=tk.NORMAL)
            text.delete(1.0, tk.END)
            text.insert(tk.END, f"User: {user_input}\nAI: {partial_response}\n\n")
            text.see(tk.END)
            text.config(state=tk.DISABLED)
            text.after(10, lambda: type_character(i + 1))

    type_character(0)

def update_text_widget(text, user_input, response):
    text.config(state=tk.NORMAL)
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"User: {user_input}\nAI: {response}\n\n")
    text.see(tk.END)
    text.config(state=tk.DISABLED)

def on_button_click(entry, text, loading_label):
    user_input = entry.get()

    loading_label.config(text="Processing...")

    ai_thread = threading.Thread(target=generate_response, args=(entry, text, loading_label, user_input))
    ai_thread.start()

    check_thread_status(entry, loading_label, ai_thread)

def check_thread_status(entry, loading_label, ai_thread):
    if ai_thread.is_alive():
        entry.after(100, lambda: check_thread_status(entry, loading_label, ai_thread))
    else:
        loading_label.config(text="")

def refresh_gui(entry, text, loading_label):
    entry.delete(0, tk.END)
    text.config(state=tk.NORMAL)
    text.delete(1.0, tk.END)
    text.config(state=tk.DISABLED)
    loading_label.config(text="")

def main():
    window = tk.Tk()
    window.title("Chat with AI")

    entry = tk.Entry(window, width=80)
    entry.pack(pady=10)

    button = tk.Button(window, text="Send", command=lambda: on_button_click(entry, text, loading_label))
    button.pack(pady=10)

    refresh_button = tk.Button(window, text="Refresh", command=lambda: refresh_gui(entry, text, loading_label))
    refresh_button.pack(pady=5)

    loading_label = tk.Label(window, text="", font=("Arial", 12))
    loading_label.pack(pady=5)

    text = tk.Text(window, height=20, width=80, state=tk.DISABLED)
    text.pack(padx=10, pady=10)

    window.mainloop()

if __name__ == "__main__":
    main()
