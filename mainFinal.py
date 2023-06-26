import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import os


def analyze_sentiment():
    # Get text from the input field
    text = input_text.get("1.0", "end-1c")

    # Preprocessing
    lower_case = text.lower()
    cleaned_text = lower_case.translate(
        str.maketrans('', '', string.punctuation))
    tokenized_words = word_tokenize(cleaned_text, "english")

    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)

    # Lemmatization
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)

    # Extract emotions
    emotion_list = []
    emotions_file = os.path.join(os.path.dirname(__file__), 'emotions.csv')
    with open(emotions_file, 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(
                ",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word in lemma_words:
                emotion_list.append(emotion)

    # Sentiment analysis
    sentiment_analyze(cleaned_text)

    # Enable the plot button
    plot_button.config(state="normal")

    # Store the emotion list for plotting
    window.emotion_list = emotion_list


def sentiment_analyze(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        sentiment_label.config(text="Negative Sentiment", fg="red")
    elif score['neg'] < score['pos']:
        sentiment_label.config(text="Positive Sentiment", fg="green")
    else:
        sentiment_label.config(text="Neutral Sentiment", fg="blue")


def plot_graph():
    # Get the stored emotion list
    emotion_list = window.emotion_list

    w = Counter(emotion_list)
    emotion_counts = pd.Series(w)
    emotion_counts.sort_values(ascending=False, inplace=True)

    # Generate x-axis tick positions using NumPy
    x_ticks = np.arange(len(emotion_counts))

    fig, ax1 = plt.subplots()
    ax1.bar(x_ticks, emotion_counts.values)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(emotion_counts.index, rotation=45, ha='right')

    ax1.set_xlabel('Emotion')
    ax1.set_ylabel('Count')
    ax1.set_title('Emotion Distribution')

    plt.tight_layout()
    plt.savefig('graph.png')
    plt.show()


# Create the GUI
window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("400x400")
# Add this line to prevent window from maximizing
window.resizable(False, False)

# Text input frame
frame1 = tk.Frame(window)
frame1.pack(pady=20)

input_label = tk.Label(frame1, text="Enter text to analyze:")
input_label.pack()

input_text = tk.Text(frame1, height=8, width=40)
input_text.pack(pady=10)

analyze_button = tk.Button(
    frame1, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack(pady=10)

plot_button = tk.Button(frame1, text="Plot Graph", command=plot_graph)
plot_button.pack(pady=10)

plot_button.config(state="disabled")

frame2 = tk.Frame(window)
frame2.pack(pady=20)

result_label = tk.Label(frame2, text="Sentiment Result:")
result_label.pack()

sentiment_label = tk.Label(frame2, text="", font=("Arial", 14, "bold"))
sentiment_label.pack()

window.mainloop()
