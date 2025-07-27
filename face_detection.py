import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Mood Mate", layout="centered")
st.title("Mood Mate")

# Load model once
@st.cache_resource
def load_emotion_model():
    return load_model("model.h5")

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Spotify Setup from secrets
client_id = st.secrets["SPOTIPY_CLIENT_ID"]
client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# Tabs
tab1, tab2 = st.tabs(['Overview', 'Model'])

with tab1:
    st.subheader("Welcome to Mood Mate")
    st.write("""
    Mood Mate – Emotion-Based Activity & Music Recommender

    🌟 Features:
    🎥 Upload an image, and our AI will detect your mood.
    💡 Get mood-based task suggestions (via LLaMA3).
    🎵 Receive Spotify playlists tailored to your emotions.
    """)

with tab2:
    st.subheader("Upload your image to detect emotion")

    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected in image. Try another one.")
        else:
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face)
            emotion_acc = emotion_labels[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100

            st.success(f"I am {round(confidence, 2)}% sure you're feeling **{emotion_acc}**.")

            # LLaMA3 prompt
            if st.button("Suggest tasks"):
                prompt_map = {
                    "Happy": "I am feeling happy. Recommend 5 fun or productive things I can do.",
                    "Sad": "I’m feeling sad. Suggest 5 comforting tasks or uplifting activities.",
                    "Angry": "I’m feeling angry. Suggest 5 ways to release this anger constructively.",
                    "Disgust": "I’m feeling disgusted. Suggest 5 things to reset my mood.",
                    "Fear": "I’m feeling anxious. Recommend 5 calming and reassuring activities.",
                    "Surprise": "I’m surprised. Suggest 5 tasks to process this feeling positively.",
                    "Neutral": "I feel neutral. Suggest 5 light tasks to explore."
                }
                with st.spinner("Getting your recommendations..."):
                    response = requests.post("http://localhost:11434/api/generate", json={
                        "model": "llama3",
                        "prompt": prompt_map[emotion_acc],
                        "stream": False
                    })
                    result = response.json()
                    st.markdown(result['response'])

            if st.button("Suggest Spotify Playlists"):
                mood_query = {
                    "Happy": "feel good",
                    "Sad": "sad songs",
                    "Angry": "anger release",
                    "Disgust": "calm down",
                    "Fear": "soothing music",
                    "Surprise": "unexpected vibes",
                    "Neutral": "chill mood"
                }.get(emotion_acc, "mood")

                try:
                    results = sp.search(q=mood_query, type='playlist', limit=5)
                    for playlist in results['playlists']['items']:
                        st.markdown(f"### [{playlist['name']}]({playlist['external_urls']['spotify']})")
                        if playlist['images']:
                            st.image(playlist['images'][0]['url'], width=300)
                except Exception as e:
                    st.error("Spotify playlist fetch failed.")
                    st.code(str(e))
