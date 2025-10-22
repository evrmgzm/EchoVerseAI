# app.py (Gelişmiş Gemini API ile RAG versiyonu - Dark Mode)
import streamlit as st
import os
import re
import torch
from dotenv import load_dotenv

# LangChain bileşenleri
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Diğer kütüphaneler
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

# --- 1. PROJE AYARLARI ---
load_dotenv()
st.set_page_config(page_title="EchoVerse AI", layout="wide", page_icon="🎥")

# --- DARK MODE CSS ---
st.markdown("""
<style>
    /* Ana tema */
    .stApp {
        background: #0a0e27;
    }
    
    /* Ana başlık */
    .main-header {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1b3d 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: #e0e7ff;
        margin-bottom: 2rem;
        border: 1px solid #3730a3;
        box-shadow: 0 8px 32px rgba(79, 70, 229, 0.15);
    }
    
    .main-header h1 {
        color: #818cf8;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #a5b4fc;
        font-size: 1.1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1629;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #818cf8;
    }
    
    /* Video bilgi kutusu */
    .video-info {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a5f 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: #e0e7ff;
        margin: 1rem 0;
        border: 1px solid #3b82f6;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
    }
    
    .video-info strong {
        color: #60a5fa;
    }
    
    /* Kaynak kutuları */
    .source-box {
        background: #1e293b;
        border-left: 4px solid #8b5cf6;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.8rem;
        border: 1px solid #2d3748;
    }
    
    .source-title {
        color: #a78bfa;
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .source-content {
        color: #cbd5e1;
        font-style: italic;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    
    /* Info kutusu */
    .info-box {
        background: #1e293b;
        border-left: 4px solid #06b6d4;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #0e7490;
        color: #e0e7ff;
    }
    
    .info-box h3 {
        color: #22d3ee;
        margin-bottom: 1rem;
    }
    
    .info-box ul {
        color: #cbd5e1;
    }
    
    /* Butonlar */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
    }
    
    /* Chat mesajları */
    .stChatMessage {
        background: #1e293b;
        border-radius: 12px;
        border: 1px solid #2d3748;
    }
    
    /* Text input */
    .stTextInput>div>div>input {
        background: #1e293b;
        color: #e0e7ff;
        border: 1px solid #3730a3;
        border-radius: 8px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 1px #6366f1;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e293b;
        color: #a78bfa;
        border-radius: 8px;
        border: 1px solid #2d3748;
    }
    
    /* Divider */
    hr {
        border-color: #2d3748;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Ana başlık
st.markdown("""
<div class="main-header">
    <h1>🎥 EchoVerse AI</h1>
    <p>YouTube Videolarıyla Akıllı Sohbet Platformu</p>
</div>
""", unsafe_allow_html=True)

# --- 2. ÇEKİRDEK FONKSİYONLAR ---

@st.cache_resource
def load_embeddings():
    """ Embedding modelini yükler ve cache'ler. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': device}
    )

def extract_video_id(url: str) -> str:
    """ YouTube URL'sinden video ID'sini ayıklar. """
    patterns = [
        r'(?:v=|\/|embed\/|watch\?v=|\&v=)([0-9A-Za-z_-]{11})', 
        r'youtu\.be\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def process_youtube_video(url: str):
    """
    Videonun transkriptini ve başlığını alır.
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Geçersiz YouTube URL'si.")
            return None, None

        st.info(f"📥 Video ID: '{video_id}' için transkript alınıyor...")
        
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['tr', 'en'])
        transcript_text = " ".join([item.text for item in transcript_data])
        
        st.success("✅ Transkript başarıyla alındı!")

        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            video_title = yt.title
        except Exception:
            video_title = f"Video (ID: {video_id})"

        return transcript_text, video_title
        
    except Exception as e:
        st.error(f"❌ Transkript alınamadı. Lütfen videonun herkese açık olduğundan ve altyazısı olduğundan emin olun.\n\n**Hata:** {e}")
        return None, None

def build_rag_chain(text: str):
    """ Gemini API ile gelişmiş RAG zincirini oluşturur. """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # --- ÖZEL PROMPT TEMPLATE ---
    prompt_template = """Sen EchoVerse AI asistanısın. Görüşün, verilen YouTube video transkriptine dayalı olarak soruları yanıtlamaktır.

KURALLAR:
1. Sadece sağlanan video içeriğinden bilgi kullan
2. Video içeriğinde olmayan bilgiler sorulursa, nazikçe "Bu bilgi videoda geçmiyor" diye belirt
3. Cevaplarını net, anlaşılır ve dostça bir dilde ver
4. Eğer video içeriği soruyla alakalı değilse, bunu açıkça söyle
5. Spekülasyon yapma, sadece video içeriğine sadık kal
6. Ancak, genel bilgiler veya açıklamalar istenmişse (örneğin bir kavramın açıklaması), kısa ve öz şekilde yardımcı olabilirsin

Bağlam (Video İçeriği):
{context}

Soru: {question}

Cevap:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # --- GOOGLE GEMINI FLASH ---
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

# --- 3. SESSION STATE ---
if "chain" not in st.session_state: 
    st.session_state.chain = None
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []
if "video_title" not in st.session_state: 
    st.session_state.video_title = ""

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### 📹 Video Yükleme Paneli")
    st.markdown("---")
    
    youtube_url = st.text_input(
        "YouTube Video URL'si",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Transkripti olan herhangi bir YouTube videosu"
    )

    if st.button("🚀 Videoyu İşle ve Başla", type="primary", use_container_width=True):
        if not youtube_url:
            st.error("⚠️ Lütfen bir YouTube URL'si girin!")
        else:
            with st.spinner("⏳ Video işleniyor..."):
                transcript, title = process_youtube_video(youtube_url)

            if transcript:
                st.session_state.video_title = title
                with st.spinner("🧠 Yapay zeka modeli hazırlanıyor..."):
                    st.session_state.chain = build_rag_chain(transcript)
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.success("✅ Sistem hazır!")
                    st.balloons()
            else:
                st.error("❌ Video işlenemedi.")
    
    st.markdown("---")
    
    # Video bilgisi
    if st.session_state.chain:
        st.markdown(f"""
        <div class="video-info">
            <strong>📺 Aktif Video:</strong><br>
            {st.session_state.video_title}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Oturumu Sıfırla", use_container_width=True):
            st.session_state.chain = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.video_title = ""
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Nasıl Kullanılır?
    1. YouTube video linkini yapıştırın
    2. "Videoyu İşle" butonuna tıklayın
    3. Video hakkında soru sorun
    4. Kaynakları görüntüleyin
    """)

# --- 5. ANA SOHBET ALANI ---
if not st.session_state.chain:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Hoş Geldiniz!</h3>
        <p>Başlamak için soldaki panelden bir YouTube video linki girin ve işleyin.</p>
        <p><strong>Özellikler:</strong></p>
        <ul>
            <li>🎯 Video içeriğine özel sorular sorun</li>
            <li>📝 Cevapların kaynaklarını görün</li>
            <li>💬 Doğal dil ile sohbet edin</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    # Önceki mesajları göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Eğer kaynak bilgisi varsa göster
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Cevabın Kaynakları (Video İçeriği)", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-title">📄 Kaynak {i}</div>
                            <div class="source-content">"{source}"</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Yeni soru girişi
    if prompt := st.chat_input("💬 Video hakkında bir soru sorun..."):
        # Kullanıcı mesajını kaydet
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan cevabı
        with st.chat_message("assistant"):
            with st.spinner("🤔 Düşünüyor..."):
                response = st.session_state.chain({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                source_docs = response.get("source_documents", [])
                
                # Cevabı göster
                st.markdown(answer)
                
                # Kaynakları işle ve göster
                if source_docs:
                    sources = [doc.page_content[:500] for doc in source_docs[:3]]
                    
                    with st.expander("📚 Cevabın Kaynakları (Video İçeriği)", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-title">📄 Kaynak {i}</div>
                                <div class="source-content">"{source}..."</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Mesajı kaynaklarla birlikte kaydet
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                
                # Sohbet geçmişini güncelle
                st.session_state.chat_history.append((prompt, answer))