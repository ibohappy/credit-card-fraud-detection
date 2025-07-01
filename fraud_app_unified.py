"""
🛡️ Kredi Kartı Dolandırıcılık Tespit Sistemi
TEK DOSYADA - Hem normal kullanıcılar hem teknik kullanıcılar için
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import random
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🛡️ Kredi Kartı Güvenlik Sistemi",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
.main-header {font-size: 2.5rem; color: #1f77b4; font-weight: bold; text-align: center;}
.sub-header {font-size: 1.5rem; color: #ff7f0e; font-weight: bold;}
.success-box {background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 1rem; border-radius: 0.5rem;}
.danger-box {background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Model yükleme
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()

# Ana başlık
st.markdown('<h1 class="main-header">🛡️ Kredi Kartı Güvenlik Kontrol Sistemi</h1>', unsafe_allow_html=True)

if model is None:
    st.error("🚨 Model dosyaları bulunamadı! Ana analizi çalıştırın: `python fraud_detection.py`")
    st.stop()

# Sidebar - Kullanıcı Seviyesi Seçimi
with st.sidebar:
    st.markdown("### 👤 Kullanıcı Seviyesi Seçin")
    user_mode = st.radio(
        "",
        options=["🏠 Normal Kullanıcı (Basit)", "🔬 Teknik Kullanıcı (Gelişmiş)"],
        help="Normal: Anlaşılır interface\nTeknik: V özellikleri dahil detaylı analiz"
    )
    
    st.markdown("---")
    st.markdown("### 📚 Sistem Hakkında")
    
    if "Normal" in user_mode:
        st.info("""
        **Bu sistem nedir?**
        Kredi kartı işlemlerinizin güvenli olup olmadığını kontrol eden yapay zeka sistemi.
        
        **Nasıl çalışır?**
        • İşlem bilgilerinizi analiz eder
        • Yapay zeka ile risk hesaplar  
        • %87 doğrulukla sonuç verir
        """)
    else:
        st.info("""
        **Teknik Detaylar:**
        • Random Forest Modeli
        • F1-Score: 0.874
        • Precision: 0.941
        • Recall: 0.816
        • 30 özellik (Time, V1-V28, Amount)
        """)
    
    # V özellikleri açıklama
    if "Teknik" in user_mode:
        with st.expander("❓ V Özellikleri Nedir?"):
            st.markdown("""
            **V1-V28**: PCA ile dönüştürülmüş özellikler
            
            **Neden V özellikleri var?**
            • 🔒 **Gizlilik**: Orijinal müşteri bilgileri korunur
            • 📉 **Boyut Azaltma**: Yüzlerce özellik 28'e indirilmiş
            • 🎯 **Performans**: Daha hızlı ve etkili
            
            **Değer Aralığı:**
            • Normal: -3 ile +3 arası
            • Tipik: -1 ile +1 arası  
            • Aşırı: ±5'e yakın (şüpheli)
            """)
    
    st.warning("**Not:** Demo amaçlıdır. Gerçek bankalar daha kapsamlı veri kullanır.")

# V özellikleri simülasyon fonksiyonu
def simulate_v_features(transaction_type, location_type, frequency, amount, time_seconds):
    """Normal kullanıcı girişlerinden V özelliklerini simüle et"""
    v_features = [0.0] * 28
    
    # İşlem türüne göre
    type_effects = {
        "Market/Bakkal": [0.1, -0.2, 0.0, 0.1, -0.1],
        "Restoran/Kafe": [0.2, 0.1, -0.1, 0.0, 0.1], 
        "Online Alışveriş": [0.5, 0.3, 0.2, -0.1, 0.4],
        "ATM Para Çekme": [-0.3, 0.4, 0.1, 0.2, -0.2],
        "Benzin İstasyonu": [0.0, -0.1, 0.2, 0.1, 0.0],
        "Eczane/Sağlık": [-0.1, 0.0, -0.2, 0.1, -0.1],
        "Eğlence/Sinema": [0.3, 0.2, 0.1, -0.1, 0.2],
        "Diğer": [0.1, 0.1, 0.0, 0.0, 0.1]
    }
    
    # Lokasyona göre
    location_effects = {
        "Yaşadığınız şehir": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Farklı şehir (Türkiye)": [0.3, 0.2, 0.1, 0.0, 0.1],
        "Yurt dışı": [1.5, 1.2, 0.8, 0.5, 1.0],
        "Online/İnternet": [0.4, 0.3, 0.2, 0.1, 0.3]
    }
    
    # Sıklığa göre 
    freq_effects = {
        "Günlük (her gün)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Haftalık (haftada birkaç kez)": [0.1, 0.1, 0.0, 0.0, 0.1],
        "Aylık (ayda birkaç kez)": [0.3, 0.2, 0.2, 0.1, 0.2],
        "Nadir (çok az yaparım)": [0.8, 0.6, 0.4, 0.3, 0.5]
    }
    
    # Miktar etkisi
    if amount > 5000:
        amount_effect = [1.2, 0.8, 0.6, 0.4, 0.9]
    elif amount > 1000:
        amount_effect = [0.6, 0.4, 0.3, 0.2, 0.4]
    elif amount > 100:
        amount_effect = [0.2, 0.1, 0.1, 0.0, 0.1]
    else:
        amount_effect = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Zaman etkisi (gece işlemleri riskli)
    if time_seconds < 21600:  # Gece
        time_effect = [0.5, 0.3, 0.2, 0.1, 0.3]
    else:
        time_effect = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # İlk 5 V özelliğini etkilerle hesapla
    for i in range(5):
        v_features[i] = (type_effects[transaction_type][i] + 
                       location_effects[location_type][i] +
                       freq_effects[frequency][i] +
                       amount_effect[i] +
                       time_effect[i])
    
    # Diğer V özelliklerini rastgele ama makul değerlerle
    for i in range(5, 28):
        v_features[i] = random.uniform(-0.5, 0.5)
    
    return v_features

# NORMAL KULLANICI INTERFACE
if "Normal" in user_mode:
    st.markdown("### 🎯 Kendi işlemlerinizi kontrol edin ve dolandırıcılık riskini öğrenin!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 İşlem Bilgilerinizi Girin</h2>', unsafe_allow_html=True)
        
        # Miktar
        amount = st.number_input(
            "💳 İşlem Miktarı (TL)", 
            min_value=0.01, 
            max_value=100000.0, 
            value=150.0,
            help="Ne kadar para harcadınız?"
        )
        
        # Zaman
        transaction_time = st.selectbox(
            "⏰ Gün Saati",
            options=[
                "Sabah (06:00-12:00)", 
                "Öğle (12:00-18:00)",
                "Akşam (18:00-24:00)",
                "Gece (00:00-06:00)"
            ],
            index=0,
            help="İşlemi hangi saatte yaptınız?"
        )
        
        # İşlem türü
        transaction_type = st.selectbox(
            "🏪 İşlem Türü",
            options=[
                "Market/Bakkal", 
                "Restoran/Kafe",
                "Online Alışveriş",
                "ATM Para Çekme",
                "Benzin İstasyonu",
                "Eczane/Sağlık",
                "Eğlence/Sinema",
                "Diğer"
            ],
            help="Neye para harcadınız?"
        )
        
        location_type = st.selectbox(
            "🌍 Lokasyon",
            options=[
                "Yaşadığınız şehir",
                "Farklı şehir (Türkiye)",
                "Yurt dışı",
                "Online/İnternet"
            ],
            help="İşlemi nerede yaptınız?"
        )
        
        frequency = st.selectbox(
            "📊 Bu Tür Harcama Sıklığınız",
            options=[
                "Günlük (her gün)",
                "Haftalık (haftada birkaç kez)", 
                "Aylık (ayda birkaç kez)",
                "Nadir (çok az yaparım)"
            ],
            help="Bu tür harcamaları ne sıklıkla yaparsınız?"
        )

    with col2:
        st.markdown('<h2 class="sub-header">🔍 Güvenlik Analizi</h2>', unsafe_allow_html=True)
        
        # Hızlı test butonları
        st.markdown("#### ⚡ Hızlı Test Örnekleri")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("✅ Güvenli İşlem Örneği", use_container_width=True):
                st.session_state.quick_test = "safe"
        
        with col_btn2:
            if st.button("🚨 Riskli İşlem Örneği", use_container_width=True):
                st.session_state.quick_test = "risky"
        
        st.markdown("---")
        
        if st.button("🛡️ GÜVENLİK KONTROLÜ YAP", type="primary", use_container_width=True):
            
            # Zaman dönüştürme
            time_mapping = {
                "Sabah (06:00-12:00)": random.randint(21600, 43200),
                "Öğle (12:00-18:00)": random.randint(43200, 64800),
                "Akşam (18:00-24:00)": random.randint(64800, 86400),
                "Gece (00:00-06:00)": random.randint(0, 21600)
            }
            time_seconds = time_mapping[transaction_time]
            
            # Hızlı test kontrolü
            if hasattr(st.session_state, 'quick_test'):
                if st.session_state.quick_test == "safe":
                    amount = 67.88
                    time_seconds = 43200
                    v_features = [0.0] * 28
                elif st.session_state.quick_test == "risky":
                    amount = 1.00
                    time_seconds = 3600
                    v_features = [-3.2, 2.8, -2.5, 3.1, -1.8, 2.9, -2.7, 1.4, -3.2, 2.3,
                                -2.8, 3.6, -3.0, 2.7, -1.4, 2.3, -2.9, 3.2, -2.1, 1.5,
                                -2.6, 3.2, -2.1, 2.9, -1.3, 2.8, -2.5, 3.0]
                del st.session_state.quick_test
            else:
                v_features = simulate_v_features(transaction_type, location_type, frequency, amount, time_seconds)
            
            # Feature array oluştur
            features = np.array([time_seconds] + v_features + [amount]).reshape(1, -1)
            
            # Normalize et ve tahmin yap
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            # Sonuçları göster
            st.markdown("---")
            st.markdown("### 📊 Analiz Sonucu")
            
            if prediction == 0:
                st.markdown(f"""
                <div class="success-box">
                <h3>✅ İşlem GÜVENLİ</h3>
                <p><strong>Risk Seviyesi:</strong> %{(probability[1]*100):.1f}</p>
                <p>Bu işlem normal görünüyor ve dolandırıcılık riski düşük.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-box">
                <h3>🚨 İşlem RİSKLİ</h3>
                <p><strong>Risk Seviyesi:</strong> %{(probability[1]*100):.1f}</p>
                <p>Bu işlem şüpheli görünüyor! Bankacınızla iletişime geçin.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk faktörleri
            st.markdown("#### 🔍 Risk Faktörleri")
            risk_factors = []
            
            if amount > 5000:
                risk_factors.append("💰 Yüksek işlem miktarı")
            if location_type == "Yurt dışı":
                risk_factors.append("🌍 Yurt dışı işlem")
            if frequency == "Nadir (çok az yaparım)":
                risk_factors.append("📊 Alışılmamış işlem türü")
            if transaction_time == "Gece (00:00-06:00)":
                risk_factors.append("🌙 Gece saatlerinde işlem")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("✅ Önemli risk faktörü bulunamadı")

# TEKNİK KULLANICI INTERFACE  
else:
    st.markdown("### 🔬 Gelişmiş Dolandırıcılık Tespit Sistemi")
    
    # Tab menu
    tab1, tab2, tab3 = st.tabs(["🔍 Canlı Tahmin", "📊 Model Analizi", "📈 Veri Görselleştirme"])
    
    with tab1:
        st.header("🔍 Gerçek Zamanlı Dolandırıcılık Tespiti")
        
        # Session state initialization
        if 'test_amount' not in st.session_state:
            st.session_state.test_amount = 100.0
        if 'test_time' not in st.session_state:
            st.session_state.test_time = 84692
        if 'test_v_features' not in st.session_state:
            st.session_state.test_v_features = [0.0] * 28
        
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            st.subheader("📊 İşlem Bilgileri")
            
            # Temel özellikler
            amount = st.number_input("💰 İşlem Miktarı ($)", 0.0, 25000.0, 
                                    value=st.session_state.test_amount, step=10.0)
            
            time_val = st.number_input("⏰ Zaman (saniye)", 0, 172792, 
                                      value=st.session_state.test_time, step=1000)
            
            # Session state güncelle
            st.session_state.test_amount = amount
            st.session_state.test_time = time_val
            
            # V özellikleri
            with st.expander("🔧 V Özellikleri (PCA Dönüştürülmüş)", expanded=False):
                st.markdown("""
                **Normal Kullanım:** Tüm V değerlerini 0.0 bırakabilirsiniz (ortalama profil)
                
                **Değer Rehberi:**
                - **-1 ile +1**: Tipik değerler
                - **-3 ile +3**: Normal aralık  
                - **±5'e yakın**: Aşırı/şüpheli değerler
                """)
                
                # V özelliklerini 4 kolon halinde düzenle
                for row in range(7):
                    cols = st.columns(4)
                    for col_idx in range(4):
                        feature_idx = row * 4 + col_idx + 1
                        if feature_idx <= 28:
                            with cols[col_idx]:
                                val = st.slider(f'V{feature_idx}', -5.0, 5.0, 
                                              value=st.session_state.test_v_features[feature_idx-1], 
                                              step=0.1, key=f'v{feature_idx}')
                                st.session_state.test_v_features[feature_idx-1] = val
            
            # Hızlı Test Butonları
            st.subheader("⚡ Hızlı Test Senaryoları")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("✅ Normal"):
                    st.session_state.test_amount = 67.88
                    st.session_state.test_time = 84692
                    st.session_state.test_v_features = [0.0] * 28
                    st.rerun()
            with col2:
                if st.button("⚠️ Şüpheli"):
                    st.session_state.test_amount = 2125.87
                    st.session_state.test_time = 45000
                    st.session_state.test_v_features = [-1.2, 0.8, -0.5, 1.1, -0.3] + [0.0] * 23
                    st.rerun()
            with col3:
                if st.button("🚨 Fraud"):
                    st.session_state.test_amount = 1.00
                    st.session_state.test_time = 100000
                    st.session_state.test_v_features = [-3.2, 2.8, -2.5, 3.1, -1.8] + [random.uniform(-2, 2) for _ in range(23)]
                    st.rerun()
            with col4:
                if st.button("🎲 Random"):
                    st.session_state.test_amount = round(random.uniform(1, 5000), 2)
                    st.session_state.test_time = random.randint(0, 172800)
                    st.session_state.test_v_features = [round(random.uniform(-3, 3), 2) for _ in range(28)]
                    st.rerun()
        
        with col_result:
            if st.button("🔍 DOLANDIRICILIK ANALİZİ YAP", type="primary", use_container_width=True):
                
                # Feature array oluştur
                features = np.array([st.session_state.test_time] + st.session_state.test_v_features + [st.session_state.test_amount]).reshape(1, -1)
                
                # Normalize et ve tahmin yap
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                
                # Sonuçları göster
                st.markdown("### 📊 Analiz Sonucu")
                
                # Tahmin sonucu
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    if prediction == 0:
                        st.success("✅ Normal İşlem")
                    else:
                        st.error("🚨 Dolandırıcılık")
                
                with col_pred2:
                    st.metric("🎯 Dolandırıcılık Riski", f"%{probability[1]*100:.2f}")
                
                # Detaylı analiz
                with st.expander("🔬 Detaylı Analiz", expanded=True):
                    st.write("**Tahmin Olasılıkları:**")
                    st.write(f"• Normal İşlem: %{probability[0]*100:.2f}")
                    st.write(f"• Dolandırıcılık: %{probability[1]*100:.2f}")
                    
                    st.write("**Kullanılan Özellikler:**")
                    st.write(f"• Zaman: {st.session_state.test_time:,} saniye")
                    st.write(f"• Miktar: ${st.session_state.test_amount:,.2f}")
                    
                    # Önemli V özelliklerini göster
                    important_vs = [(16, 'V17'), (13, 'V14'), (11, 'V12'), (9, 'V10'), (15, 'V16')]
                    st.write("**Önemli V Özellikleri:**")
                    for idx, name in important_vs:
                        st.write(f"• {name}: {st.session_state.test_v_features[idx]:.2f}")
                
                # Risk gösterge çubuğu
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability[1]*100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Dolandırıcılık Risk Seviyesi (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 Model Performans Analizi")
        
        # Model metrikleri
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 F1-Score", "0.874")
        with col2:
            st.metric("🔍 Precision", "0.941")
        with col3:
            st.metric("📈 Recall", "0.816")
        with col4:
            st.metric("⚖️ Accuracy", "0.999")
        
        st.info("""
        **Model Detayları:**
        - **Algoritma:** Random Forest
        - **Özellik Sayısı:** 30 (Time + V1-V28 + Amount)  
        - **Eğitim Verisi:** 284,807 işlem
        - **İmbalance Oranı:** 577:1 (Normal:Dolandırıcılık)
        """)
    
    with tab3:
        st.header("📈 Veri Görselleştirme")
        
        # Örnek veri yükle
        try:
            df = pd.read_csv('creditcard.csv')
            
            # Özet istatistikler
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Toplam İşlem", f"{len(df):,}")
            with col2:
                st.metric("Normal İşlem", f"{(df['Class']==0).sum():,}")
            with col3:
                st.metric("Dolandırıcılık", f"{(df['Class']==1).sum():,}")
            
            # Class dağılımı
            class_counts = df['Class'].value_counts()
            fig = px.pie(values=class_counts.values, names=['Normal', 'Dolandırıcılık'], 
                        title="Normal vs Dolandırıcılık İşlem Oranı")
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("📁 creditcard.csv dosyası bulunamadı.")

# Footer
st.markdown("---")
st.markdown("### 🔗 Sistem Bilgileri")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Model:** Random Forest")
with col2:
    st.info("**Doğruluk:** %87.4")
with col3:
    st.info("**Özellik:** 30 adet")

st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
🛡️ Bu sistem demo amaçlıdır. Gerçek bankacılık sistemleri daha kapsamlı güvenlik önlemleri kullanır.
</div>
""", unsafe_allow_html=True) 