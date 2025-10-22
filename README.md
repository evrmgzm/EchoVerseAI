---
title: EchoVerse AI
emoji: 🎥
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
python_version: 3.11
app_file: app.py
---
# EchoVerse AI 🤖🎥: YouTube Videolarıyla Etkileşimli Sohbet

Bu proje, kullanıcılara herhangi bir YouTube videosunun içeriği hakkında sohbet etme imkanı sunan, RAG (Retrieval-Augmented Generation) tabanlı bir chatbot uygulamasıdır.

---

## 📜 Projenin Amacı

Günümüzde video içerikleri, bilgi edinmenin en popüler yollarından biridir. Ancak, uzun videoları izlemek veya belirli bir bilgiyi bulmak için videonun tamamını taramak zaman alıcı olabilir. **EchoVerse AI**, bu soruna bir çözüm sunar.

Projenin temel amacı, kullanıcının sağladığı bir YouTube videosunun transkriptini temel alarak bir bilgi tabanı oluşturmak ve bu bilgi tabanı üzerinden soruları yanıtlayan akıllı bir asistan geliştirmektir. Bu sayede kullanıcılar, videoları izlemeden içerikleri hakkında hızlıca bilgi edinebilir, özetler alabilir veya spesifik sorularına anında yanıt bulabilirler.

## 📊 Veri Seti

Bu projede statik bir veri seti kullanılmamaktadır. Bunun yerine, uygulamanın bilgi tabanı **dinamik olarak ve gerçek zamanlı** bir şekilde oluşturulur.

*   **Veri Kaynağı:** Kullanıcının girdiği herhangi bir herkese açık YouTube videosu.
*   **Veri Toplama Metodolojisi:**
    1.  Kullanıcı, arayüz üzerinden bir YouTube URL'si girer.
    2.  `pytube` kütüphanesi ile videonun başlığı gibi temel meta verileri alınır.
    3.  `youtube-transcript-api` kütüphanesi kullanılarak videonun Türkçe veya İngilizce altyazı transkripti çekilir.
    4.  Elde edilen bu transkript metni, projenin o anki oturum için "veri setini" ve bilgi kaynağını oluşturur.

Bu yaklaşım, uygulamanın herhangi bir video hakkında anında uzmanlaşabilmesini sağlayarak esnek ve güçlü bir yapı sunar.

## 🛠️ Kullanılan Yöntemler ve Çözüm Mimarisi

Proje, modern bir **Retrieval-Augmented Generation (RAG)** mimarisi üzerine kurulmuştur. Bu mimari, büyük dil modellerinin (LLM) yeteneklerini, harici ve güncel bir bilgi tabanıyla birleştirerek daha doğru ve bağlama uygun cevaplar üretmesini sağlar.

### RAG Akış Şeması

```
[YouTube URL] -> [Transkript Çekme] -> [Metin Parçalama (Chunking)] -> [Vektör Temsilleri (Embeddings)] -> [Vektör Veritabanı (FAISS)]
                                                                                                               ^
                                                                                                               | (Arama ve Benzerlik Bulma)
                                                                                                               |
                                     [Kullanıcı Sorusu] -> [Embedding] -> [Benzerlik Araması] -> [Alakalı Parçalar] -> [LLM'e Gönderme (Gemini)] -> [Cevap]
```

### Teknoloji Stack'i

*   **Generation Model (Üretici Model):** **Google Gemini Flash** (`models/gemini-flash-latest`). Hızı ve performansı sayesinde akıcı bir sohbet deneyimi sunmak için tercih edilmiştir.
*   **Embedding Model (Vektör Temsil Modeli):** **Hugging Face - `sentence-transformers/all-MiniLM-L6-v2`**. Açık kaynaklı, hızlı ve etkili bir model olup, metin parçalarını anlamlı vektörlere dönüştürmek için kullanılmıştır.
*   **Vektör Veritabanı:** **FAISS (Facebook AI Similarity Search)**. Metin parçalarının vektör temsillerini bellekte (in-memory) verimli bir şekilde saklamak ve anlamsal arama yapmak için kullanılmıştır.
*   **RAG Pipeline Framework:** **LangChain**. Veri yükleme, parçalama, embedding oluşturma, arama ve LLM ile etkileşim gibi tüm RAG adımlarını düzenleyen ve birbirine bağlayan ana çerçevedir. Projede `ConversationalRetrievalChain` kullanılarak sohbet geçmişi hafızası da sağlanmıştır.
*   **Web Arayüzü:** **Streamlit**. Hızlı ve kolay bir şekilde interaktif bir web uygulaması geliştirmek için kullanılmıştır.

## 📈 Elde Edilen Sonuçlar

Proje sonucunda aşağıdaki hedeflere ulaşılmıştır:
*   Kullanıcı dostu ve interaktif bir web arayüzü başarıyla geliştirilmiştir.
*   Herhangi bir YouTube videosu için anlık olarak RAG tabanlı bir sohbet asistanı oluşturma yeteneği kazanılmıştır.
*   Sohbet geçmişini hatırlayan (`ConversationalRetrievalChain` sayesinde), bağlama duyarlı ve akıcı bir diyalog akışı sağlanmıştır.
*   Google Gemini ve Hugging Face gibi güçlü ve modern yapay zeka modelleri başarılı bir şekilde entegre edilmiştir.

## 💻 Kurulum ve Çalışma Kılavuzu

Projenin yerel makinenizde çalıştırılması için aşağıdaki adımları takip edebilirsiniz.

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/evrmgzm/EchoVerseAI.git
cd EchoVerseAI
```

### 2. Sanal Ortam Oluşturun ve Aktif Edin
```bash
# Sanal ortamı oluşturun
python -m venv venv

# Sanal ortamı aktif edin
# Windows için:
.\venv\Scripts\activate
# macOS/Linux için:
source venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
Proje için gerekli tüm kütüphaneler `requirements.txt` dosyasında listelenmiştir.
```bash
pip install -r requirements.txt
```

### 4. API Anahtarlarını Ayarlayın
Projenin çalışabilmesi için bir Google Gemini API anahtarına ihtiyacınız vardır.
*   Projenin ana dizininde `.env` adında bir dosya oluşturun.
*   Dosyanın içine aşağıdaki gibi API anahtarınızı ekleyin:
    ```
    GEMINI_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

### 5. Uygulamayı Çalıştırın
Tüm adımlar tamamlandıktan sonra aşağıdaki komut ile Streamlit uygulamasını başlatabilirsiniz.
```bash
streamlit run app.py
```
Uygulama, varsayılan olarak tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

## 🌐 Web Arayüzü ve Kullanım Kılavuzu

Uygulama arayüzü basit ve kullanıcı dostu olacak şekilde tasarlanmıştır.

1.  **URL Girişi:** Sol taraftaki kenar çubuğuna, hakkında sohbet etmek istediğiniz YouTube videosunun linkini yapıştırın.
2.  **İşlemi Başlatma:** "🚀 Videoyu İşle" butonuna tıklayın. Bu aşamada uygulama, videonun transkriptini alacak ve RAG mimarisini hazırlayacaktır.
3.  **Sohbete Başlama:** Hazırlık tamamlandığında, ekranın alt kısmında yer alan sohbet kutusuna video ile ilgili sorularınızı yazarak sohbete başlayabilirsiniz.
![WhatsApp Görsel 2025-10-22 saat 23 33 35_e7e486e6](https://github.com/user-attachments/assets/f2d561c3-19f8-456c-8f69-d756236778f6)
---

![WhatsApp Görsel 2025-10-22 saat 23 34 22_be67bf2e](https://github.com/user-attachments/assets/649b0940-363a-4732-91a4-275df4fa98e0)

---

![WhatsApp Görsel 2025-10-22 saat 23 34 50_3ef054a7](https://github.com/user-attachments/assets/e028af76-7612-4017-8d21-44d506e3c303)

---

![WhatsApp Görsel 2025-10-22 saat 23 35 08_23fa3dbd](https://github.com/user-attachments/assets/5ae0c969-edad-45b3-9249-c2bfbf006a8b)

---

![WhatsApp Görsel 2025-10-22 saat 23 36 50_6edbf79c](https://github.com/user-attachments/assets/e7b81153-f723-4f1f-8c45-af6e74582680)

---


---

### 🚀 Deploy Edilen Uygulama

Uygulamanın canlı versiyonuna aşağıdaki linkten ulaşabilirsiniz:

**[👉 EchoVerse AI Uygulamasına Gitmek İçin Tıklayın](https://your-streamlit-app-link.streamlit.app/)**


