import re
import pickle
import streamlit as st

# =========================
# LOAD MODEL & VECTORIZER
# =========================
@st.cache_resource
def load_files():
    with open("model_sms.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf

model, tfidf = load_files()

# =========================
# PREPROCESSING TEXT
# Sesuai notebook:
# lower, hapus URL, hapus angka/simbol, stopword, stemming
# =========================
@st.cache_resource
def load_sastrawi():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        stemmer = StemmerFactory().create_stemmer()
        stopwords_indo = StopWordRemoverFactory().get_stop_words()
        custom_stopwords = ["yg", "sih", "nya", "aja", "yth"]
        stopwords_indo.extend(custom_stopwords)

        return stemmer, set(stopwords_indo)
    except Exception:
        return None, set(["yg", "sih", "nya", "aja", "yth"])

stemmer, stopwords_indo = load_sastrawi()

def text_preprocessing(text):
    if not isinstance(text, str):
        return ""

    # lowercasing
    text = text.lower()

    # hapus URL/link
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # hapus angka, tanda baca, simbol
    text = re.sub(r"[^a-z\s]", " ", text)

    # hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    cleaned_words = []

    for word in words:
        if word not in stopwords_indo:
            if stemmer is not None:
                word = stemmer.stem(word)
            cleaned_words.append(word)

    return " ".join(cleaned_words)

# =========================
# LABEL
# =========================
label_map = {
    0: "Normal",
    1: "Penipuan",
    2: "Spam/Promo"
}

# =========================
# TAMPILAN WEB
# =========================
st.set_page_config(
    page_title="Klasifikasi SMS",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Klasifikasi SMS Normal, Penipuan, dan Spam/Promo")
st.write("Aplikasi ini menggunakan model SVM dan TF-IDF dari notebook kamu.")

sms_input = st.text_area(
    "Masukkan isi SMS:",
    height=150,
    placeholder="Contoh: Selamat! Anda mendapatkan hadiah 10 juta, klik link berikut..."
)

if st.button("Prediksi SMS"):
    if sms_input.strip() == "":
        st.warning("Masukkan teks SMS terlebih dahulu.")
    else:
        clean_text = text_preprocessing(sms_input)
        text_tfidf = tfidf.transform([clean_text])
        pred = model.predict(text_tfidf)[0]
        hasil = label_map.get(int(pred), str(pred))

        st.subheader("Hasil Prediksi")
        if hasil == "Normal":
            st.success(f"✅ SMS ini termasuk: {hasil}")
        elif hasil == "Penipuan":
            st.error(f"⚠️ SMS ini termasuk: {hasil}")
        else:
            st.warning(f"📢 SMS ini termasuk: {hasil}")

        with st.expander("Lihat hasil preprocessing"):
            st.write(clean_text)

st.divider()
st.caption("Model: LinearSVC/SVM | Fitur: TF-IDF | Label: 0 Normal, 1 Penipuan, 2 Spam/Promo")
