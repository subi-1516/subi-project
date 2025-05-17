import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# NLTK setup
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load your dataset
df = pd.read_csv("fake_news_dataset.csv")

# Quick inspection
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Rename target column if needed
if 'label' not in df.columns:
    df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

# Preprocessing text data
def clean_text(text):
    words = str(text).lower().split()
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].map({'FAKE': 0, 'REAL': 1}) if df['label'].dtype == object else df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# GUI for interactive predictions
def predict_news():
    input_text = input_box.get("1.0", tk.END)
    clean_input = clean_text(input_text)
    input_vector = vectorizer.transform([clean_input])
    prediction = model.predict(input_vector)[0]
    result = "REAL" if prediction == 1 else "FAKE"
    messagebox.showinfo("Prediction Result", f"The news is: {result}")

# Tkinter GUI
app = tk.Tk()
app.title("Fake News Detection")
app.geometry("500x300")

tk.Label(app, text="Enter News Text:", font=("Arial", 14)).pack(pady=10)
input_box = tk.Text(app, height=8, width=60)
input_box.pack(pady=5)

tk.Button(app, text="Check News", font=("Arial", 12), bg="blue", fg="white", command=predict_news).pack(pady=10)
app.mainloop()
