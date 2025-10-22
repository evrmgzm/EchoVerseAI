# app.py (Gemini API ile RAG versiyonu)
import streamlit as st
import os
import re
import torch
from dotenv import load_dotenv

# LangChain bileşenleri
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Diğer kütüphaneler
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

# --- 1. PROJE AYARLARI ---
load_dotenv()
st.set_page_config(page_title="EchoVerse AI", layout="wide", page_icon="🎥")
st.title("EchoVerse AI: YouTube Transkriptiyle Sohbet Edin 🎥")

# --- 2. ÇEKİRDEK FONKSİYONLAR ---

@st.cache_resource
def load_embeddings():
    """ Embedding modelini yükler ve cache'ler. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})

def extract_video_id(url: str) -> str:
    """ YouTube URL'sinden video ID'sini ayıklar. """
    patterns = [r'(?:v=|\/|embed\/|watch\?v=|\&v=)([0-9A-Za-z_-]{11})', r'youtu\.be\/([0-9A-Za-z_-]{11})']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# BU BLOĞU KOPYALAYIP GEMINI'Lİ app.py DOSYANIZA YAPIŞTIRIN
# Bu, sizin Mixtral versiyonunuzda çalışan ve mevcut kütüphane
# sürümünüzle uyumlu olan koddur.

def process_youtube_video(url: str):
    """
    Dokümantasyona uygun 'fetch' metodunu kullanarak videonun transkriptini ve başlığını alır.
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Geçersiz YouTube URL'si.")
            return None, None

        st.info(f"'{video_id}' ID'li video için transkript alınıyor...")
        
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['tr', 'en'])
        transcript_text = " ".join([item.text for item in transcript_data])
        
        st.success("Transkript başarıyla alındı. Video başlığı alınıyor...")

        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            video_title = yt.title
        except Exception:
            video_title = f"Video (ID: {video_id})"

        return transcript_text, video_title
        
    except Exception as e:
        st.error(f"Transkript alınamadı. Lütfen videonun herkese açık olduğundan ve Türkçe veya İngilizce altyazısı olduğundan emin olun.\nHata Detayı: {e}")
        return None, None

def build_rag_chain(text: str):
    """ Gemini API ile RAG zincirini oluşturur. """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # --- GOOGLE GEMINI 2.5 FLASH ---
    llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest", # <--- DOĞRU MODEL ADI
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
    # max_output_tokens=1024 # Bu parametre genellikle yeni modellerde convert_system_message_to_human=True ile birlikte kullanılır, şimdilik kaldırabiliriz veya bırakabiliriz.
)
    # --------------------------------

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# --- 3. STREAMLIT ARAYÜZÜ ---

if "chain" not in st.session_state: st.session_state.chain = None
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "video_title" not in st.session_state: st.session_state.video_title = ""

with st.sidebar:
    st.header("📹 Video Ayarları")
    youtube_url = st.text_input("YouTube Video URL'si", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("🚀 Videoyu İşle", type="primary", use_container_width=True):
        if not youtube_url:
            st.error("Lütfen bir YouTube URL'si girin!")
        else:
            with st.spinner("⏳ Video transkripti alınıyor..."):
                transcript, title = process_youtube_video(youtube_url)

            if transcript:
                st.success(f"✅ Transkript bulundu!")
                st.session_state.video_title = title
                with st.spinner("🧠 Yapay zeka hazırlanıyor..."):
                    st.session_state.chain = build_rag_chain(transcript)
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.balloons()
            else:
                st.error("❌ Transkript alınamadı. Lütfen başka bir video deneyin.")

if st.session_state.chain:
    st.success(f"Sistem Hazır: **{st.session_state.video_title}**")
    if st.button("🗑️ Oturumu Sıfırla", use_container_width=True):
        st.session_state.chain = None
        st.session_state.messages = []
        st.session_state.video_title = ""
        st.rerun()

# Ana Sohbet Alanı
if not st.session_state.chain:
    st.info("👈 Başlamak için soldaki alana bir YouTube linki yapıştırın ve 'Videoyu İşle' butonuna basın.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Video hakkında bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor..."):
                response = st.session_state.chain({"question": prompt, "chat_history": st.session_state.chat_history})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_history.append((prompt, answer))
