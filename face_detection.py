import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests

import os

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="playlist-read-private"
    ))

st.set_page_config(page_title="Mood Mate", layout="centered")
st.title("Mood Mate")
tab1,tab2=st.tabs(['Overview','Model'])
with tab1:
    st.subheader("Welcome to Mood Mate")
    st.write("""
                    Mood Mate – Emotion-Based Activity & Music Recommender
    Mood Mate is a smart, interactive web application that detects your emotional state through a single image capture and offers personalized task suggestions and Spotify playlist recommendations to match your mood.

    🌟 Key Features:
    🎥 One-Click Emotion Detection:
    Using a trained deep learning model (model.h5) and your webcam, the app captures a single image and accurately identifies your emotion (e.g., Happy, Sad, Angry, etc.).

    💡 Smart Activity Recommendations:
    Once your emotion is detected, Mood Mate uses LLM-based prompting (via LLaMA3) to suggest custom tasks suited to your current emotional state — whether you're looking to calm down, stay productive, or lift your spirits.

    🎵 Mood-Based Spotify Playlists:
    Integrated with the Spotify API, the app recommends curated playlists that align with your detected mood, enhancing your emotional experience through music.

    🎯 Use Case:
    Perfect for students, professionals, or anyone looking for a quick emotional check-in and personalized support — whether through calming tasks, fun activities, or vibe-matching music.
                    """)
with  tab2:
    st.subheader("Try to click a pick of you!")
    def recommender(prompt):
        with st.spinner("Let me suggest some tasks for you"):
            response=requests.post("http://localhost:11434/api/generate",
                                        json={
                                            "model":"llama3",
                                            "prompt":prompt,
                                            "stream": False
                                        }
                                        )
            result=response.json()
            st.write(result['response'])
        
    def ask_for_song(emotion):
        st.write("Would You like me to suggest some playlists?")
        if st.button("Yes"):
            songs(emotion)

    def songs(emotion):
        st.write("Here are some Spotify playlists for your mood")

        emotion_to_query = {
                "Happy": "feel good",
                "Sad": "sad songs",
                "Angry": "anger release",
                "Disgust": "calm down",
                "Fear": "soothing music",
                "Surprise": "unexpected vibes",
                "Neutral": "chill mood"
            }

        query = emotion_to_query.get(emotion, "mood")
        
        try:
            results = sp.search(q=query, type='playlist', limit=5)
            playlists = results['playlists']['items']
            
            for playlist in playlists:
                name = playlist['name']
                url = playlist['external_urls']['spotify']
                img_url = playlist['images'][0]['url'] if playlist['images'] else None

                st.markdown(f"### [{name}]({url})")
                if img_url:
                    st.image(img_url, width=300)

        except Exception as e:
            st.error("Failed to fetch playlists from Spotify.")
            st.code(str(e))


    def emotiondetect():
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to capture image.")
            exit()

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            print("No face detected.")
            exit()

        
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]

        
        face = cv2.resize(face, (48, 48))  
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)  

        if model.input_shape[-1] == 3:
            face = np.repeat(face, 3, axis=-1)  

            pred = model.predict(face)
            emotion = emotion_labels[np.argmax(pred)]
            confidence = np.max(pred)
            return emotion,confidence
        

        model = load_model("model.h5")

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        if  st.button("Capture image"):
        
            emotion_acc,confident=emotiondetect()
            confident=confident*100
            st.success(f"Hmm I seem {round(confident,2)} sure that your mood is {emotion_acc}")
            prompt=""
            if emotion_acc=='Happy':
                prompt="""I am feeling happy and upbeat! Recommend 5
                fun or productive things I can do to make the most of this mood."""
                recommender(prompt)
                
            elif emotion_acc=='Sad':
                prompt="""I’m feeling sad. Suggest 5 gentle, 
                comforting tasks or uplifting activities I can do to feel a bit better."""
                recommender(prompt)
                
            elif emotion_acc=='Angry':
                prompt="""I’m feeling angry right now. Suggest 5 healthy
                ways to release this anger or shift
                my focus. Maybe some physical activities, creative outlets,
                or calming techniques."""
                recommender(prompt)
                
            elif emotion_acc=='Disgust':
                prompt="""I’m feeling disgusted or uncomfortable. Suggest 5 ways to reset my
                mood or distract myself with something refreshing and uplifting."""
                recommender(prompt)
                
            elif emotion_acc=='Fear':
                prompt="""I’m feeling anxious or afraid. Recommend 5 tasks that can calm 
                me down or make me feel safe and in control."""
                recommender(prompt)
                
            elif emotion_acc=='Surprise':
                prompt="""I’m feeling surprised or caught off guard.
                Suggest 5 tasks that help me process this unexpected feeling—positively 
                or calmly."""
                recommender(prompt)
                
            elif emotion_acc=='Neutral':
                prompt="""I feel neutral or in-between emotions. Suggest 5 tasks
                that are mildly engaging or help me discover what I actually want to do."""
                recommender(prompt)
                
            else:
                st.warning("Oh Oh some problem occurred!")
            ask_for_song(emotion_acc)
            
