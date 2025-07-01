# 🛡️ AI Kredi Kartı Dolandırıcılık Tespit Sistemi

**12 AI Modeli ile Gerçek Zamanlı Finansal Güvenlik Çözümü**

## 🎯 **Proje Amacı**

Bu proje, **12 farklı yapay zeka modeli** kullanarak kredi kartı işlemlerinde dolandırıcılığı gerçek zamanlı olarak tespit eden gelişmiş bir sistem sunar. Bankalar ve finans kurumları için tasarlanan bu sistem, **%87.43 F1-Score** ile şüpheli işlemleri anlık olarak belirleyerek finansal kayıpları önlemeyi amaçlar.

### 🚨 **Çözmeye Çalıştığı Problem**
- **Dolandırıcılık** yıllık milyarlarca dolar zarar veriyor
- **Geleneksel yöntemler** yavaş ve yetersiz kalıyor  
- **Manuel kontrol** insan hatasına açık
- **False positive** oranları müşteri memnuniyetsizliği yaratıyor

### 💡 **Çözüm Yaklaşımı**
- **12 farklı AI modeli** kapsamlı test edildi
- **Context7 modern tasarım** ile profesyonel görselleştirme
- **Imbalanced data** problemi profesyonelce çözüldü
- **Modern web arayüzü** ile kullanıcı dostu deneyim
- **Gerçek zamanlı analiz** (2.34 saniyede sonuç)

## 🏆 **Teknik Başarımlar**

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **F1-Score** | **87.43%** | Dolandırıcılık tespit başarı oranı |
| **Precision** | **94.12%** | Yanlış alarm oranı düşük |
| **Recall** | **81.63%** | Gerçek dolandırıcılıkları yakalama |
| **ROC-AUC** | **95.33%** | Model güvenilirlik skoru |
| **Analiz Hızı** | **2.34 saniye** | Gerçek zamanlı performans |
| **Test Edilen Model** | **12 AI Modeli** | Kapsamlı karşılaştırma |

## 🤖 **Test Edilen 12 AI Modeli**

### **🏅 Geleneksel Machine Learning (8 Model)**
1. **🏆 Random Forest** - **F1: 0.874** ⭐ **(Kazanan Model)**
2. **🥈 XGBoost** - **F1: 0.806** (Hızlı ve güvenilir)
3. **🥉 Decision Tree** - **F1: 0.811** (Yorumlanabilir)
4. **Logistic Regression** - **F1: 0.720** (Baseline)
5. **Linear SVM** - **F1: 0.690** (Optimize edilmiş)
6. **K-Nearest Neighbors** - **F1: 0.650** (Instance-based)
7. **LightGBM** - **F1: 0.635** (Hafif model)
8. **Naive Bayes** - **F1: 0.612** (Probabilistic)

### **🧠 Deep Learning (3 Model)**
9. **Neural Network (32-16)** - **F1: 0.819** (Balanced architecture)
10. **Neural Network (64-32-16)** - **F1: 0.801** (Deep architecture)
11. **Neural Network (128-64)** - **F1: 0.785** (Wide architecture)

### **🔍 Anomaly Detection (1 Model)**
12. **Isolation Forest** - **F1: 0.682** (Unsupervised learning)

## 📊 **Kullanılan Veri Seti**

Bu proje **Kaggle'ın ünlü Credit Card Fraud Detection** veri setini kullanır:

🔗 **Veri Kaynağı:** [Kaggle - Credit Fraud Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/input)

### 📈 **Veri Seti Özellikleri:**
- **284,807 kredi kartı işlemi** (gerçek veriler)
- **492 dolandırıcılık vakası** (%0.17)
- **284,315 normal işlem** (%99.83)
- **28 PCA dönüştürülmüş özellik** (V1-V28) - gizlilik korumalı
- **Time, Amount** özellikleri

> ⚠️ **Not:** `creditcard.csv` dosyası (150MB) GitHub'ın 100MB limitini aştığı için bu repository'de bulunmaz. Yukarıdaki Kaggle linkinden indirebilirsiniz.

## 🎨 **Context7 Modern Görselleştirme**

### 📈 **Yeni Görselleştirme Özellikleri:**
- **🏆 Model Performance Ranking** - 12 AI modelinin F1-Score karşılaştırması
- **📊 Precision vs Recall Analysis** - Detaylı performans analizi
- **🔥 ROC-AUC Heatmap** - Model güvenilirlik haritası
- **⚡ Confusion Matrix** - Detaylı hata analizi
- **🎯 Feature Importance** - En önemli özelliklerin analizi
- **📉 Model Category Breakdown** - ML/DL/Anomaly sınıflandırması

### 🖼️ **Profesyonel PNG Çıktıları:**
- `model_results.png` - 12 AI modelinin kapsamlı karşılaştırması
- `feature_importance.png` - Özellik önem analizi
- `data_analysis.png` - Veri seti keşif analizi
- `clustering_analysis.png` - Veri kümesi dağılım analizi

## 🔧 **Teknik Mimari**

### ⚖️ **Imbalanced Data Çözümleri:**
- **SMOTE** (Synthetic oversampling)
- **NearMiss** (Undersampling)
- **Original Data** ⭐ (En iyi sonuç)

### 📊 **Feature Engineering:**
- **RobustScaler** ile outlier-resistant scaling
- **PCA preserved** V1-V28 features
- **Temporal analysis** için Time feature
- **Amount normalization** 

## 🚀 **Kurulum ve Çalıştırma**

### 1. **Repository'yi Clone Edin**
```bash
git clone https://github.com/ibohappy/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. **Gerekli Kütüphaneleri Kurun**
```bash
pip install -r requirements.txt
```

### 3. **Veri Setini İndirin**
[Kaggle linkinden](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/input) `creditcard.csv` dosyasını indirip proje klasörüne koyun.

### 4. **Model Eğitimi (Opsiyonel)**
```bash
python fraud_detection.py
```

### 5. **Web Uygulamasını Başlatın**
```bash
streamlit run fraud_app_streamlit.py
```

### 🌐 **Veya Direkt Canlı Demo'yu Deneyin!**
**Demo Linki:** https://credit-card-fraud-detection-12models.streamlit.app/

## 🌐 **Canlı Demo**

**🚀 Uygulamayı Canlı Deneyin:** 
- **Web Demo:** https://credit-card-fraud-detection-12models.streamlit.app/
- **Lokal:** `http://localhost:8501`

> 🎯 Canlı demo'da 12 AI modelinin karşılaştırmasını görebilir, V1-V28 parametrelerini ayarlayarak gerçek zamanlı dolandırıcılık tespiti yapabilirsiniz!

## 📱 **Web Uygulaması Özellikleri**

### 🎯 **Ana Dashboard**
- **12 AI Model Analysis** - Kapsamlı model karşılaştırması
- **Real-time metrics** gösterimi
- **Context7 modern tasarım** - Profesyonel görselleştirme
- **Risk analizi** grafikleri
- **Sistem durumu** monitoring

### 🔧 **Dolandırıcılık Tespiti**
- **V1-V28 parametreleri** (PCA features)
- **İşlem miktarı** ve **zaman** ayarları
- **Tab-based interface** (Temel, Gelişmiş, Uzman, Pro)
- **Rastgele test** örnekleri
- **Anlık risk skoru** hesaplama

### 📊 **Raporlama & Analiz**
- **12 AI Model** detaylı performans analizi
- **Görselleştirme galeri** - PNG çıktıları
- **Model seçim** rehberi
- **Performance monitoring**
- **Teknik metrikler** dashboard

## 🎨 **Modern UI/UX Tasarımı**

### 🌟 **Kullanılan Teknolojiler:**
- **Streamlit** - Web framework
- **streamlit-shadcn-ui** - Modern components
- **Context7 Design System** - Professional styling
- **Plotly** - İnteraktif grafikler
- **Custom CSS** - Advanced styling

### 🎭 **Tasarım Özellikleri:**
- **Context7 modern bileşenler**
- **Dark/Light mode** support
- **Responsive design** (mobil uyumlu)
- **Professional color scheme**
- **Intuitive navigation**
- **Real-time feedback**

## 📈 **Kapsamlı Model Karşılaştırma Sonuçları**

### 🏆 **En İyi Performans Gösteren Modeller**

| Sıra | Model | F1-Score | Precision | Recall | ROC-AUC | Kategori |
|------|-------|----------|-----------|---------|---------|----------|
| 🥇 | **Random Forest** | **0.8743** | **0.9412** | **0.8163** | **0.9533** | **Geleneksel ML** |
| 🥈 | **Neural Network (32-16)** | **0.8191** | **0.8556** | **0.7857** | **-** | **Deep Learning** |
| 🥉 | **Decision Tree** | **0.8111** | **0.8902** | **0.7449** | **0.8095** | **Geleneksel ML** |
| 4 | **XGBoost** | **0.8063** | **0.8280** | **0.7857** | **0.9256** | **Geleneksel ML** |
| 5 | **Neural Network (64-32-16)** | **0.8015** | **0.8345** | **0.7703** | **-** | **Deep Learning** |

### 📊 **Model Kategori Analizi**

**🤖 Geleneksel ML Ortalaması:** F1-Score: 0.728  
**🧠 Deep Learning Ortalaması:** F1-Score: 0.802  
**🔍 Anomaly Detection:** F1-Score: 0.682  

### 🎯 **Önerilen Kullanım Senaryoları**

| Model | Önerilen Kullanım | Güçlü Yanları | Zayıf Yanları |
|-------|------------------|---------------|---------------|
| **Random Forest** | **Production** ⭐ | Yüksek doğruluk, stabil | Yavaş training |
| **Neural Network** | Research & Development | Adaptif öğrenme | Complex setup |
| **XGBoost** | High-speed processing | Hızlı, optimize edilmiş | Parameter tuning |
| **Decision Tree** | Explainable AI | Yorumlanabilir | Overfitting riski |

## 🔮 **Gelecek Geliştirmeler**

### 🚀 **Teknik İyileştirmeler**
- [ ] **Graph Neural Networks** için network analysis
- [ ] **Ensemble methods** ile model kombinasyonu
- [ ] **Real-time API** development
- [ ] **GPU acceleration** için CUDA support
- [ ] **AutoML** ile otomatik model seçimi
- [ ] **Transformer models** entegrasyonu

### 🌐 **Platform Genişletmeleri**
- [ ] **Mobile app** development
- [ ] **API Gateway** integration
- [ ] **Cloud scaling** (AWS/Azure)
- [ ] **Blockchain** transaction support
- [ ] **Multi-language** support
- [ ] **Enterprise dashboard**

### 📊 **Business Intelligence**
- [ ] **Advanced dashboard** (Tableau/PowerBI)
- [ ] **A/B testing** framework
- [ ] **Customer behavior** analysis
- [ ] **Risk scoring** algorithms
- [ ] **Compliance reporting**
- [ ] **Real-time alerts** system

## 🏢 **Gerçek Dünya Uygulamaları**

### 🏦 **Bankalar**
- **Kredi kartı** işlem monitoring
- **ATM** güvenlik sistemi
- **Online banking** fraud prevention
- **Mobile payment** security

### 💳 **Fintech Şirketleri**
- **Digital wallet** protection
- **P2P transfer** security
- **Cryptocurrency** exchange monitoring
- **Investment platform** safeguards

### 🛒 **E-ticaret**
- **Online payment** fraud detection
- **Marketplace** seller verification
- **Subscription** abuse prevention
- **Chargeback** reduction

## 📄 **Lisans**

Bu proje **MIT License** altında lisanslanmıştır. Ticari kullanım için serbesttir.

## 🤝 **Katkıda Bulunma**

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 🚀 **Projeyi Hemen Deneyin!**

**🌐 Canlı Demo:** https://credit-card-fraud-detection-12models.streamlit.app/

**🎯 Bu proje, 12 farklı AI modeli ile finans sektöründe yapay zeka kullanımının gücünü gösteren production-ready bir çözümdür. Random Forest modeli %87.43 F1-Score ile kazanan performans sergileyerek bankalar ve fintech şirketleri için gerçek zamanlı dolandırıcılık tespiti sağlar.** 

**⭐ Beğendiyseniz star vermeyi unutmayın!**

---

### 📈 **Son Güncellemeler**
- ✅ **12 AI Modelinin** kapsamlı analizi tamamlandı
- ✅ **Context7 modern tasarım** entegre edildi  
- ✅ **Professional PNG** görselleştirmeler eklendi
- ✅ **Streamlit Cloud** deployment güncellemesi
- ✅ **Random Forest** kazanan model olarak belirlendi (%87.43 F1-Score)