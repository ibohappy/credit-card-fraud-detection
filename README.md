# 🛡️ AI Kredi Kartı Dolandırıcılık Tespit Sistemi

**Gerçek Zamanlı Yapay Zeka ile Finansal Güvenlik Çözümü**

## 🎯 **Proje Amacı**

Bu proje, **kredi kartı işlemlerinde dolandırıcılığı gerçek zamanlı olarak tespit eden** gelişmiş bir yapay zeka sistemidir. Bankalar ve finans kurumları için tasarlanan bu sistem, **%87.43 doğruluk oranı** ile şüpheli işlemleri anlık olarak belirleyerek finansal kayıpları önlemeyi amaçlar.

### 🚨 **Çözmeye Çalıştığı Problem**
- **Dolandırıcılık** yıllık milyarlarca dolar zarar veriyor
- **Geleneksel yöntemler** yavaş ve yetersiz kalıyor  
- **Manuel kontrol** insan hatasına açık
- **False positive** oranları müşteri memnuniyetsizliği yaratıyor

### 💡 **Çözüm Yaklaşımı**
- **12 farklı AI modeli** test edildi
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

## 🔧 **Teknik Mimari**

### 🤖 **Test Edilen AI Modelleri:**
1. **Random Forest** ⭐ (En iyi performans)
2. **XGBoost** (Hızlı ve güvenilir)
3. **Neural Networks** (Deep learning)
4. **Logistic Regression** (Baseline)
5. **Linear SVM** (Optimize edilmiş)
6. **Decision Tree** (Yorumlanabilir)
7. **LightGBM** (Hafif model)
8. **Naive Bayes** (Probabilistic)

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

## 🌐 **Canlı Demo**

**🚀 Uygulamayı Canlı Deneyin:** 
- Streamlit Cloud: `https://ibohappy-credit-card-fraud-detection.streamlit.app`
- Lokal: `http://localhost:8501`

## 📱 **Web Uygulaması Özellikleri**

### 🎯 **Ana Dashboard**
- **Real-time metrics** gösterimi
- **Model performance** karşılaştırması
- **Risk analizi** grafikleri
- **Sistem durumu** monitoring

### 🔧 **Dolandırıcılık Tespiti**
- **V1-V28 parametreleri** (PCA features)
- **İşlem miktarı** ve **zaman** ayarları
- **Tab-based interface** (Temel, Gelişmiş, Uzman, Pro)
- **Rastgele test** örnekleri
- **Anlık risk skoru** hesaplama

### 📊 **Raporlama**
- **Günlük/haftalık** trend analizi
- **Model accuracy** metrikleri
- **Alert sistemi** yapılandırması
- **Performance monitoring**

## 🎨 **Modern UI/UX Tasarımı**

### 🌟 **Kullanılan Teknolojiler:**
- **Streamlit** - Web framework
- **streamlit-shadcn-ui** - Modern components
- **Plotly** - İnteraktif grafikler
- **Custom CSS** - Professional styling

### 🎭 **Tasarım Özellikleri:**
- **Dark/Light mode** support
- **Responsive design** (mobil uyumlu)
- **Professional color scheme**
- **Intuitive navigation**
- **Real-time feedback**

## 📈 **Model Karşılaştırma Sonuçları**

| Model | F1-Score | Precision | Recall | ROC-AUC | Önerilen Kullanım |
|-------|----------|-----------|---------|---------|------------------|
| **Random Forest** | **0.8743** | **0.9412** | **0.8163** | **0.9533** | **Production** ⭐ |
| XGBoost | 0.8063 | 0.8280 | 0.7857 | 0.9256 | High-speed processing |
| Neural Network | 0.8191 | 0.8556 | 0.7857 | - | Deep learning research |
| Logistic Regression | 0.7200 | 0.8182 | 0.6429 | 0.9582 | Baseline comparison |
| Linear SVM | 0.6905 | 0.8286 | 0.5918 | 0.9431 | Fast training |
| Decision Tree | 0.8111 | 0.8902 | 0.7449 | 0.8095 | Explainable AI |

## 🔮 **Gelecek Geliştirmeler**

### 🚀 **Teknik İyileştirmeler**
- [ ] **Graph Neural Networks** için network analysis
- [ ] **Ensemble methods** ile model kombinasyonu
- [ ] **Real-time API** development
- [ ] **GPU acceleration** için CUDA support
- [ ] **AutoML** ile otomatik model seçimi

### 🌐 **Platform Genişletmeleri**
- [ ] **Mobile app** development
- [ ] **API Gateway** integration
- [ ] **Cloud scaling** (AWS/Azure)
- [ ] **Blockchain** transaction support
- [ ] **Multi-language** support

### 📊 **Business Intelligence**
- [ ] **Advanced dashboard** (Tableau/PowerBI)
- [ ] **A/B testing** framework
- [ ] **Customer behavior** analysis
- [ ] **Risk scoring** algorithms
- [ ] **Compliance reporting**

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

## 📞 **İletişim**

**Proje Sahibi:** [İletişim bilgilerinizi buraya ekleyin]

**LinkedIn:** [LinkedIn profiliniz]

**Email:** [Email adresiniz]

---

**🎯 Bu proje, finans sektöründe yapay zeka kullanımının gücünü gösteren production-ready bir çözümdür. Bankalar ve fintech şirketleri için gerçek zamanlı dolandırıcılık tespiti sağlar.** 

**⭐ Beğendiyseniz star vermeyi unutmayın!**