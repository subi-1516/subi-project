# Save this as app.py
# streamlit run app.py
import streamlit as st

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = rf_model.predict(vec)[0]
    return "Fake" if pred == 1 else "Real"

st.title("Fake News Detector")
user_input = st.text_area("Enter News Text:")
if st.button("Predict"):
    result = predict_news(user_input)
    st.subheader(f"The news is: {result}")
