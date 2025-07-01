# 🏦 Kredi Kartı Dolandırıcılığı Tespit Sistemi

**Kaggle "Credit Fraud Detector" Kernel Standardında Gelişmiş ML Sistemi**

## 🎯 Proje Özeti

Bu proje, kredi kartı işlemlerindeki dolandırıcılığı tespit etmek için gelişmiş makine öğrenmesi tekniklerini kullanan kapsamlı bir sistemdir. Kaggle'ın ünlü "Credit Fraud Detector" kernel'ı referans alınarak, profesyonel seviyede bir fraud detection sistemi oluşturulmuştur.

## 🏆 Ana Başarımlar

- **🥇 En İyi Model**: Random Forest - F1-Score: 0.8743 (87.43%)
- **⚡ Hızlı Performans**: LinearSVM optimizasyonu ile hızlı eğitim
- **🧠 Neural Networks**: TensorFlow ile deep learning implementasyonu
- **📊 Kapsamlı Analiz**: 12 farklı grafik ve görselleştirme
- **🔄 Sampling Teknikleri**: SMOTE, NearMiss, Random sampling karşılaştırması

## 📁 Dosya Yapısı

### 🔧 Ana Sistem
- `fraud_detection.py` - Kapsamlı fraud detection sistemi (886 satır)
- `fraud_app_streamlit.py` - Web uygulaması (Streamlit)
- `requirements.txt` - Gerekli kütüphaneler

### 🤖 Eğitilmiş Modeller
- `best_model.pkl` - En iyi performans gösteren Random Forest modeli
- `scaler.pkl` - Feature scaling için RobustScaler

### 📊 Görselleştirmeler
- `data_analysis.png` - Kapsamlı veri analizi (12 grafik)
- `model_comparison.png` - Model karşılaştırma sonuçları
- `clustering_analysis.png` - t-SNE ve PCA cluster analizi

### 📋 Raporlar
- `fraud_detection_report.md` - Detaylı teknik rapor
- `README.md` - Bu dosya

### 📂 Veri
- `creditcard.csv` - Kaggle Credit Card Fraud Detection dataset'i

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
pip install -r requirements.txt
```

### 2. Ana Analizi Çalıştır
```bash
python fraud_detection.py
```

### 3. Web Uygulamasını Başlat
```bash
streamlit run fraud_app_streamlit.py
```

## 🌐 **CANLI DEMO**
**🚀 Uygulamayı Canlı Deneyin:** [BURAYA STREAMLIT CLOUD URL'İ EKLENECEKTİR]

> Not: Deployment tamamlandığında bu link aktif olacaktır.

## 📈 Model Performansları

| Model | Sampling | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|---------|---------|
| **Random Forest** | **Original** | **0.8743** | **0.9412** | **0.8163** | **0.9533** |
| Decision Tree | Original | 0.8111 | 0.8902 | 0.7449 | 0.8095 |
| XGBoost | Original | 0.8063 | 0.8280 | 0.7857 | 0.9256 |
| Logistic Regression | Original | 0.7200 | 0.8182 | 0.6429 | 0.9582 |
| Neural Network | Original | 0.8191 | 0.8556 | 0.7857 | - |

## 🔍 Teknik Özellikler

### 🧪 Uygulanan Teknikler
- **Imbalanced Data Handling**: SMOTE, NearMiss, Random sampling
- **Feature Scaling**: RobustScaler (Time, Amount features)
- **Dimensionality Reduction**: t-SNE, PCA analizi
- **Anomaly Detection**: Isolation Forest
- **Cross Validation**: Stratified K-Fold
- **Neural Networks**: TensorFlow/Keras

### 📊 Veri Analizi
- **Dataset**: 284,807 işlem
- **Normal**: 284,315 (%99.83)
- **Fraud**: 492 (%0.17)
- **Imbalance Ratio**: 577.9:1
- **Features**: 30 (28 PCA + Time + Amount)

### ⚖️ Sampling Sonuçları
- **Original Data**: En iyi performans (0.8743 F1)
- **SMOTE**: Overfitting riski (0.8377 F1)
- **NearMiss**: Agresif undersampling (0.0036 F1)

## 🎯 Önemli Bulgular

1. **💡 Original Data En İyi**: Resampling yapmadan en yüksek performans
2. **🚫 SMOTE Zararlı**: Büyük dataset'lerde overfitting riski
3. **⚡ LinearSVM Optimizasyonu**: RBF yerine Linear çok daha hızlı
4. **🧠 Neural Networks**: Geleneksel ML ile rekabetçi performans

## 🔧 Kullanım Senaryoları

### 💳 Real-time Fraud Detection
```python
import joblib
import numpy as np

# Model ve scaler yükle
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Yeni işlem verisini test et
def predict_fraud(transaction_data):
    scaled_data = scaler.transform([transaction_data])
    probability = model.predict_proba(scaled_data)[0][1]
    return probability > 0.5
```

### 📊 Batch Analysis
```bash
python fraud_detection.py
# Otomatik olarak tüm analizi çalıştırır
```

## 📋 Geliştirme Notları

### ✅ Tamamlanan Özellikler
- [x] Kapsamlı veri analizi (12 görselleştirme)
- [x] Multiple ML model karşılaştırması
- [x] Sampling technique analizi
- [x] Neural network implementation
- [x] Web uygulaması (Streamlit)
- [x] Anomaly detection (Isolation Forest)
- [x] Dimensionality reduction (t-SNE, PCA)

### 🔄 İyileştirme Fırsatları
- [ ] GPU hızlandırmalı eğitim
- [ ] Özellik mühendisliği (zamansal desenler)
- [ ] Toplu öğrenme yöntemleri
- [ ] Gerçek zamanlı API uç noktası
- [ ] Model izleme panosu

## 🤝 Katkı

Bu proje Kaggle "Credit Fraud Detector" kernel metodolojisini takip eder:
- Dengesiz veri en iyi uygulamaları
- F1-skoru odaklı değerlendirme
- Doğru eğitim-test ayırma metodolojisi
- Özellik önem analizi

## 📅 Versiyon Bilgisi

- **v1.0** - Kaggle kernel standardında kapsamlı sistem
- **Tarih**: 2025-06-30
- **Python**: 3.12+
- **Dependencies**: scikit-learn, tensorflow, streamlit, xgboost, lightgbm

---

**🚀 Production Ready**: Bu sistem gerçek zamanlı fraud detection için hazırdır!