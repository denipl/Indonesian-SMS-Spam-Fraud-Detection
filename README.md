# Web Klasifikasi SMS

Aplikasi web ini dibuat dari:
- `model_sms.pkl`
- `tfidf_vectorizer.pkl`

## Cara menjalankan

1. Buka folder ini di VS Code
2. Install library:

```bash
pip install -r requirements.txt
```

3. Jalankan web:

```bash
streamlit run app.py
```

4. Masukkan teks SMS, lalu klik **Prediksi SMS**.

## Label Prediksi

- `0` = Normal
- `1` = Penipuan
- `2` = Spam/Promo
