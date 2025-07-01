"""
🛡️ AI Güvenlik Merkezi - Premium Edition
Modern Streamlit Shadcn UI ile Profesyonel Dolandırıcılık Tespit Sistemi
"""

import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import random
from datetime import datetime, timedelta
import os

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🛡️ AI Güvenlik Merkezi | Premium Edition",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS - Minimal ve Clean
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    .main .block-container {
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
        max-width: 1400px;
    }
    
    /* Header styling */
    .ai-security-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .ai-security-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .ai-security-header p {
        font-size: 1.2rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Custom styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-active { background: #dcfce7; color: #166534; }
    .status-ready { background: #fef3c7; color: #92400e; }
    .status-test { background: #dbeafe; color: #1e40af; }
    .status-low { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="ai-security-header">
    <h1>🛡️ AI Güvenlik Merkezi</h1>
    <p>Premium Edition - Gelişmiş Dolandırıcılık Tespit Sistemi</p>
</div>
""", unsafe_allow_html=True)

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("Model dosyaları bulunamadı! Lütfen model.pkl ve scaler.pkl dosyalarının mevcut olduğundan emin olun.")
        return None, None

# Ana dashboard
def main_dashboard():
    st.markdown('<div class="section-header">📊 Sistem Performans Özeti</div>', unsafe_allow_html=True)
    
    # Gerçek zamanlı durum göstergesi
    status_cols = st.columns([3, 1])
    with status_cols[0]:
        st.markdown("### 🔴 Canlı Durum Monitörü")
    with status_cols[1]:
        if ui.button("🔄 Yenile", key="refresh_btn"):
            st.rerun()
    
    # Performans metrikleri - 4 sütunlu grid
    cols = st.columns(4)
    
    # Dinamik veriler (simüle edilmiş)
    total_transactions = random.randint(280000, 290000)
    detected_frauds = random.randint(480, 520)
    accuracy = round(random.uniform(99.85, 99.95), 2)
    
    with cols[0]:
        ui.metric_card(
            title="Toplam İşlem",
            content=f"{total_transactions:,}",
            description="Son 24 saat içinde",
            key="total_transactions"
        )
    
    with cols[1]:
        growth = round(random.uniform(0.1, 0.3), 2)
        ui.metric_card(
            title="Tespit Edilen Dolandırıcılık",
            content=str(detected_frauds),
            description=f"+{growth}% artış",
            key="detected_fraud"
        )
    
    with cols[2]:
        ui.metric_card(
            title="Sistem Doğruluğu",
            content=f"{accuracy}%",
            description="Premium seviye",
            key="accuracy"
        )
    
    with cols[3]:
        ui.metric_card(
            title="Gerçek Zamanlı Koruma",
            content="AKTİF",
            description="7/24 çalışıyor",
            key="protection_status"
        )

    # Gerçek zamanlı işlem akışı
    st.markdown('<div class="section-header">⚡ Gerçek Zamanlı İşlem Akışı</div>', unsafe_allow_html=True)
    
    # Simulated real-time transactions
    transaction_cols = st.columns(3)
    
    for i, col in enumerate(transaction_cols):
        with col:
            transaction_type = random.choice(["Kredi Kartı", "Banka Transferi", "Online Ödeme"])
            amount = random.randint(10, 5000)
            risk_level = random.choice(["Düşük", "Orta", "Yüksek"])
            
            risk_colors = {"Düşük": "🟢", "Orta": "🟡", "Yüksek": "🔴"}
            
            with ui.card(key=f"realtime_transaction_{i}"):
                ui.element("h4", children=[f"💳 {transaction_type}"], key=f"trans_type_{i}")
                ui.element("p", children=[f"Miktar: ${amount}"], key=f"trans_amount_{i}")
                ui.element("p", children=[f"Risk: {risk_colors[risk_level]} {risk_level}"], key=f"trans_risk_{i}")
                ui.element("small", children=["Az önce"], key=f"trans_time_{i}")

    # Model Performans Tablosu
    st.markdown('<div class="section-header">🤖 AI Model Performans Analizi</div>', unsafe_allow_html=True)
    
    # Tabs for different views
    table_tab = ui.tabs(
        options=['📊 Performans Tablosu', '📈 Görsel Analiz', '⚙️ Model Ayarları'],
        default_value='📊 Performans Tablosu',
        key="model_analysis_tabs"
    )
    
    if table_tab == '📊 Performans Tablosu':
        # Model performance data
        model_data = [
            {
                "Model": "🌲 Rastgele Orman",
                "F1-Skoru": 0.874,
                "Kesinlik": 0.941,
                "Duyarlılık": 0.816,
                "Doğruluk": 0.999,
                "Durum": "🥇 AKTİF"
            },
            {
                "Model": "🚀 XGBoost",
                "F1-Skoru": 0.806,
                "Kesinlik": 0.828,
                "Duyarlılık": 0.786,
                "Doğruluk": 0.998,
                "Durum": "🥈 HAZIR"
            },
            {
                "Model": "⚡ LightGBM",
                "F1-Skoru": 0.798,
                "Kesinlik": 0.821,
                "Duyarlılık": 0.775,
                "Doğruluk": 0.998,
                "Durum": "🥉 HAZIR"
            },
            {
                "Model": "📈 Logistic Regression",
                "F1-Skoru": 0.712,
                "Kesinlik": 0.768,
                "Duyarlılık": 0.662,
                "Doğruluk": 0.997,
                "Durum": "📋 TEST"
            },
            {
                "Model": "🧠 Naive Bayes",
                "F1-Skoru": 0.045,
                "Kesinlik": 0.024,
                "Duyarlılık": 0.831,
                "Doğruluk": 0.835,
                "Durum": "❌ DÜŞÜK"
            }
        ]
        
        model_df = pd.DataFrame(model_data)
        ui.table(data=model_df, maxHeight=400)
        
        # Model selection
        st.markdown("**🎯 Model Seçimi**")
        selected_model = ui.select(
            options=["Rastgele Orman", "XGBoost", "LightGBM", "Logistic Regression", "Naive Bayes"],
            key="model_selector"
        )
        if selected_model:
            st.success(f"Seçili Model: {selected_model}")
    
    elif table_tab == '📈 Görsel Analiz':
        # Model comparison chart
        models = ["Rastgele Orman", "XGBoost", "LightGBM", "Logistic Reg.", "Naive Bayes"]
        f1_scores = [0.874, 0.806, 0.798, 0.712, 0.045]
        accuracy_scores = [0.999, 0.998, 0.998, 0.997, 0.835]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='F1-Skoru',
            x=models,
            y=f1_scores,
            marker_color='#667eea'
        ))
        fig.add_trace(go.Bar(
            name='Doğruluk',
            x=models,
            y=accuracy_scores,
            marker_color='#764ba2'
        ))
        
        fig.update_layout(
            title='Model Performans Karşılaştırması',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif table_tab == '⚙️ Model Ayarları':
        settings_cols = st.columns(2)
        
        with settings_cols[0]:
            with ui.card(key="model_settings"):
                ui.element("h4", children=["⚙️ Model Konfigürasyonu"], key="model_config_title")
                
                threshold = ui.slider(
                    default_value=[0.5],
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    label="Dolandırıcılık Eşiği",
                    key="fraud_threshold"
                )[0]
                
                auto_retrain = ui.switch(
                    default_checked=True,
                    label="Otomatik Yeniden Eğitim",
                    key="auto_retrain_switch"
                )
                
                ui.element("p", children=[f"Eşik: {threshold:.2f}"], key="threshold_display")
                ui.element("p", children=[f"Otomatik Eğitim: {'Açık' if auto_retrain else 'Kapalı'}"], key="auto_retrain_display")
        
        with settings_cols[1]:
            with ui.card(key="performance_settings"):
                ui.element("h4", children=["📊 Performans Ayarları"], key="perf_settings_title")
                
                batch_size = ui.select(
                    options=["32", "64", "128", "256"],
                    key="batch_size_select"
                )
                
                learning_rate = ui.select(
                    options=["0.001", "0.01", "0.1"],
                    key="learning_rate_select"
                )
                
                ui.element("p", children=[f"Batch Size: {batch_size or '64'}"], key="batch_display")
                ui.element("p", children=[f"Learning Rate: {learning_rate or '0.01'}"], key="lr_display")

    # Sistem Bilgileri
    st.markdown('<div class="section-header">⚙️ Sistem Konfigürasyonu</div>', unsafe_allow_html=True)
    
    config_cols = st.columns(2)
    
    with config_cols[0]:
        with ui.card(key="training_params"):
            ui.element("h3", children=["🎯 Eğitim Parametreleri"], key="training_title")
            ui.element("p", children=["• Toplam İşlem: 284,807"], key="param1")
            ui.element("p", children=["• Özellik Boyutu: 30"], key="param2")
            ui.element("p", children=["• Çapraz Doğrulama: 5-katmanlı"], key="param3")
            ui.element("p", children=["• Sınıf Dengesi: 577:1"], key="param4")
    
    with config_cols[1]:
        with ui.card(key="system_requirements"):
            ui.element("h3", children=["💻 Sistem Gereksinimleri"], key="system_title")
            ui.element("p", children=["• Python: 3.8+"], key="req1")
            ui.element("p", children=["• RAM: 2GB minimum"], key="req2")
            ui.element("p", children=["• İşlemci: Çok çekirdekli"], key="req3")
            ui.element("p", children=["• Depolama: 100MB"], key="req4")
    
    # Alerts and Notifications
    st.markdown('<div class="section-header">🚨 Son Uyarılar</div>', unsafe_allow_html=True)
    
    alert_types = [
        {"type": "success", "message": "Sistem güncellemesi başarıyla tamamlandı", "time": "2 dakika önce"},
        {"type": "warning", "message": "Yüksek riskli işlem tespit edildi", "time": "5 dakika önce"},
        {"type": "info", "message": "Günlük rapor hazırlandı", "time": "1 saat önce"}
    ]
    
    for i, alert in enumerate(alert_types):
        alert_cols = st.columns([1, 4, 2])
        with alert_cols[0]:
            if alert["type"] == "success":
                st.markdown("✅")
            elif alert["type"] == "warning":
                st.markdown("⚠️")
            else:
                st.markdown("ℹ️")
        
        with alert_cols[1]:
            st.markdown(f"**{alert['message']}**")
        
        with alert_cols[2]:
            st.markdown(f"*{alert['time']}*")

    # ==================== YENİ BÖLÜM: AI MODEL GÖRSEL RAPORLARI ====================
    st.markdown('<div class="section-header">📊 AI Model Analiz Raporları</div>', unsafe_allow_html=True)
    st.markdown("*Detaylı model performansı ve veri analizi görsel raporları*")
    
    # Context7 tabs for visual reports
    visual_reports_tab = ui.tabs(
        options=['📊 Veri Analizi', '🎯 Feature Önem Sırası', '📈 Model Karşılaştırma', '🔬 Clustering Görselleştirme'],
        default_value='📊 Veri Analizi',
        key="homepage_visual_reports_tabs"
    )
    
    if visual_reports_tab == '📊 Veri Analizi':
        st.markdown("### 📊 Kapsamlı Veri Seti Analizi")
        
        try:
            import os
            image_path = os.path.join(os.getcwd(), "data_analysis.png")
            if os.path.exists(image_path):
                st.image(image_path, caption="🔍 Advanced Fraud Detection - Comprehensive Data Analysis", use_container_width=True)
                
                # Summary cards
                analysis_cols = st.columns(4)
                
                with analysis_cols[0]:
                    ui.metric_card(
                        title="Veri Seti Boyutu", 
                        content="284,807",
                        description="toplam işlem",
                        key="homepage_dataset_size"
                    )
                
                with analysis_cols[1]:
                    ui.metric_card(
                        title="Sınıf Dağılımı",
                        content="99.8% / 0.2%",
                        description="Normal / Dolandırıcılık",
                        key="homepage_class_dist"
                    )
                
                with analysis_cols[2]:
                    ui.metric_card(
                        title="Feature Sayısı",
                        content="30",
                        description="V1-V28 + Time + Amount",
                        key="homepage_feature_count"
                    )
                
                with analysis_cols[3]:
                    ui.metric_card(
                        title="Analiz Dönemi",
                        content="48 Saat",
                        description="zaman aralığı",
                        key="homepage_time_period"
                    )
                
            else:
                st.error(f"📊 data_analysis.png bulunamadı: {image_path}")
                
        except Exception as e:
            st.error(f"📊 Veri analizi görselinde hata: {str(e)}")
    
    elif visual_reports_tab == '🎯 Feature Önem Sırası':
        st.markdown("### 🎯 En Önemli 10 Feature Analizi")
        
        try:
            import os
            image_path = os.path.join(os.getcwd(), "feature_importance.png")
            if os.path.exists(image_path):
                st.image(image_path, caption="🏆 Feature Importance Rankings - V17 Leading with 18.7%", use_container_width=True)
                
                # Top features cards
                feature_cols = st.columns(3)
                
                with feature_cols[0]:
                    ui.metric_card(
                        title="🥇 En Önemli Feature",
                        content="V17",
                        description="18.7% önem skoru",
                        key="homepage_top_feature"
                    )
                
                with feature_cols[1]:
                    ui.metric_card(
                        title="🥈 İkinci Sıra",
                        content="V14", 
                        description="17.1% önem skoru",
                        key="homepage_second_feature"
                    )
                
                with feature_cols[2]:
                    ui.metric_card(
                        title="🥉 Üçüncü Sıra",
                        content="V12",
                        description="10.6% önem skoru", 
                        key="homepage_third_feature"
                    )
                
            else:
                st.error(f"🎯 feature_importance.png bulunamadı: {image_path}")
                
        except Exception as e:
            st.error(f"🎯 Feature importance görselinde hata: {str(e)}")
    
    elif visual_reports_tab == '📈 Model Karşılaştırma':
        st.markdown("### 📈 12 AI Modelinin Detaylı Performans Karşılaştırması")
        
        try:
            import os
            image_path = os.path.join(os.getcwd(), "model_results.png")
            if os.path.exists(image_path):
                st.image(image_path, caption="🤖 Comprehensive Model Performance Analysis - Random Forest Champion!", use_container_width=True)
                
                # Model performance cards
                model_perf_cols = st.columns(4)
                
                with model_perf_cols[0]:
                    ui.metric_card(
                        title="🏆 Kazanan Model",
                        content="Random Forest",
                        description="F1-Score: 0.874",
                        key="homepage_winner_model"
                    )
                
                with model_perf_cols[1]:
                    ui.metric_card(
                        title="⚡ En Hızlı",
                        content="Linear SVM",
                        description="Optimized performance",
                        key="homepage_fastest_model"
                    )
                
                with model_perf_cols[2]:
                    ui.metric_card(
                        title="🎯 En Yüksek Precision",
                        content="94.12%",
                        description="Random Forest",
                        key="homepage_highest_precision"
                    )
                
                with model_perf_cols[3]:
                    ui.metric_card(
                        title="📊 Test Edilen Model",
                        content="12",
                        description="farklı AI algoritması",
                        key="homepage_tested_models"
                    )
                
                # 12 AI Modeli detaylı açıklama
                st.markdown("### 🤖 Test Edilen 12 AI Modeli:")
                st.markdown("""
                **🎯 Geleneksel Makine Öğrenmesi (8 Model):**
                - 🌲 **Random Forest** (Winner - F1: 0.874)
                - 🚀 **XGBoost** (Runner-up - F1: 0.806)  
                - 🌳 **Decision Tree** (F1: 0.811)
                - 📈 **Logistic Regression** (F1: 0.720)
                - ⚡ **Linear SVM** (F1: 0.690)
                - 🔍 **K-Nearest Neighbors** (F1: 0.650)
                - 💡 **LightGBM** (F1: 0.404)
                - 🧠 **Naive Bayes** (F1: 0.110)
                
                **🧠 Derin Öğrenme (3 Varyant):**
                - 🤖 **Neural Network (Original)** (F1: 0.785)
                - 🤖 **Neural Network (SMOTE)** (F1: 0.798)
                - 🤖 **Neural Network (NearMiss-1)** (F1: 0.742)
                
                **🔍 Anomali Tespiti (1 Model):**
                - 🌲 **Isolation Forest** (F1: 0.342)
                
                *Toplam: 12 farklı AI algoritması 3 farklı veri dengeleme tekniği ile test edildi.*
                """)
                
            else:
                st.error(f"📈 model_results.png bulunamadı: {image_path}")
                
        except Exception as e:
            st.error(f"📈 Model karşılaştırma görselinde hata: {str(e)}")
    
    elif visual_reports_tab == '🔬 Clustering Görselleştirme':
        st.markdown("### 🔬 t-SNE ve PCA Clustering Analizi")
        
        try:
            import os
            image_path = os.path.join(os.getcwd(), "clustering_analysis.png")
            if os.path.exists(image_path):
                st.image(image_path, caption="🔬 Advanced Clustering Visualization - Clear Fraud vs Normal Separation", use_container_width=True)
                
                # Clustering analysis cards
                cluster_cols = st.columns(4)
                
                with cluster_cols[0]:
                    ui.metric_card(
                        title="🎯 t-SNE Analizi",
                        content="Net Ayrım",
                        description="Fraud vs Normal clusters",
                        key="homepage_tsne_analysis"
                    )
                
                with cluster_cols[1]:
                    ui.metric_card(
                        title="📊 PCA Variance",
                        content="69.6%",
                        description="İlk 2 bileşen (61.2% + 8.4%)",
                        key="homepage_pca_variance"
                    )
                
                with cluster_cols[2]:
                    ui.metric_card(
                        title="🔍 Pattern Ayrımı",
                        content="Belirgin",
                        description="Dolandırıcılık pattern'ları",
                        key="homepage_pattern_separation"
                    )
                
                with cluster_cols[3]:
                    ui.metric_card(
                        title="✅ Model Doğrulaması",
                        content="Görsel Kanıt",
                        description="AI başarısının ispatı",
                        key="homepage_model_validation"
                    )
                
            else:
                st.error(f"🔬 clustering_analysis.png bulunamadı: {image_path}")
                
        except Exception as e:
            st.error(f"🔬 Clustering analizi görselinde hata: {str(e)}")
    
    # Call-to-action section
    st.markdown("---")
    cta_cols = st.columns([2, 1, 2])
    
    with cta_cols[1]:
        if ui.button("🚀 Dolandırıcılık Tespitini Dene", key="homepage_try_detection"):
            st.switch_page("fraud_detection_page")  # This will be handled by the main navigation

# Dolandırıcılık Tespit Sayfası
def fraud_detection_page():
    st.markdown('<div class="section-header">🔍 Dolandırıcılık Tespit Analizi</div>', unsafe_allow_html=True)
    
    model, scaler = load_model()
    if model is None:
        st.error("Model yüklenemedi!")
        return
    
    # İşlem detayları girişi
    with ui.card(key="transaction_input"):
        ui.element("h3", children=["💳 İşlem Detayları"], key="transaction_title")
        
        # Ana parametreler
        cols = st.columns(3)
        with cols[0]:
            amount = ui.input(
                default_value="100.0",
                type='text',
                placeholder="İşlem miktarı ($)",
                key="amount_input"
            )
        
        with cols[1]:
            time_seconds = ui.input(
                default_value="3600",
                type='text',
                placeholder="Zaman (saniye)",
                key="time_input"
            )
        
        with cols[2]:
            ui.element("br", key="spacer")
            analyze_btn = ui.button("🚀 ANALİZ BAŞLAT", key="analyze_btn")
    
    # Gelişmiş Parametreler - Modern Card Tasarımı
    st.markdown('<div class="section-header">🔧 Gelişmiş AI Parametreleri (V1-V28)</div>', unsafe_allow_html=True)
    st.markdown("*Bu parametreler PCA ile dönüştürülmüş özelliklerdir ve dolandırıcılık tespitinde kritik rol oynar.*")
    
    # Quick actions (widget'lardan önce) - Context7 Direct Button Approach
    action_cols = st.columns(3)
    
    with action_cols[0]:
        reset_clicked = ui.button("🔄 Tüm Parametreleri Sıfırla", key="reset_all_params")
        if reset_clicked:
            # Session state'i tamamen temizle
            keys_to_clear = [key for key in st.session_state.keys() if key.startswith("v_param_")]
            for key in keys_to_clear:
                del st.session_state[key]
            st.success("✅ Tüm parametreler sıfırlandı!")
            st.rerun()
    
    with action_cols[1]:
        random_clicked = ui.button("🎲 Rastgele Örnek Yükle", key="load_random_sample")
        if random_clicked:
            # Session state'i temizle ve yeni rastgele değerler set et
            keys_to_clear = [key for key in st.session_state.keys() if key.startswith("v_param_")]
            for key in keys_to_clear:
                del st.session_state[key]
            
            # Rastgele değerler için session state set et
            for i in range(1, 29):
                random_val = round(random.uniform(-2.0, 2.0), 1)
                # Her tab için ayrı ayrı set et
                for tab_suffix in ["all_tab", "basic_tab", "advanced_tab", "expert_tab", "pro_tab"]:
                    key = f"v_param_{i}_{tab_suffix}"
                    st.session_state[key] = [random_val]
            
            st.success("🎲 Rastgele değerler yüklendi!")
            st.rerun()
    
    with action_cols[2]:
        # Info card hakkında parametre kullanımı
        st.markdown("**ℹ️ İpucu:**")
        st.markdown("Parametreleri ayarladıktan sonra analiz butonuna basın")
    
    v_features = {}
    
    # Parameter grupları için tabs
    param_tabs = ui.tabs(
        options=['📋 Tüm Parametreler', '🎯 Temel Parametreler (V1-V7)', '⚡ Gelişmiş (V8-V14)', '🔬 Uzman (V15-V21)', '🚀 Pro (V22-V28)'],
        default_value='📋 Tüm Parametreler',
        key="parameter_tabs"
    )
    
    if param_tabs == '📋 Tüm Parametreler':
        st.markdown("### 📋 Tüm AI Parametreleri (V1-V28)")
        st.markdown("*Tüm parametreleri tek ekranda görüntüleyip düzenleyebilirsiniz*")
        
        # 4 sütunlu grid layout
        param_cols = st.columns(4)
        
        for i in range(28):
            col_index = i % 4
            with param_cols[col_index]:
                with ui.card(key=f"all_param_card_{i+1}"):
                    # Her parametre için benzersiz ve bağımsız key
                    param_num = i + 1
                    unique_slider_key = f"v_param_{param_num}_all_tab"
                    
                    # Parametre kategorisine göre renk ve açıklama
                    if i < 7:
                        category = "🎯 Temel"
                        color_class = "text-blue-600"
                        desc = "Risk Faktörü"
                    elif i < 14:
                        category = "⚡ Gelişmiş" 
                        color_class = "text-purple-600"
                        desc = "Analiz Paterni"
                    elif i < 21:
                        category = "🔬 Uzman"
                        color_class = "text-red-600"
                        desc = "Güvenlik Skoru"
                    else:
                        category = "🚀 Pro"
                        color_class = "text-cyan-600"
                        desc = "AI Faktörü"
                    
                    ui.element("h5", children=[f"V{param_num}"], className=f"{color_class} font-bold mb-1", key=f"all_param_title_{param_num}")
                    ui.element("small", children=[f"{category} - {desc}"], className="text-gray-500 text-xs", key=f"all_param_desc_{param_num}")
                    
                    # Context7 streamlit-shadcn-ui doğru kullanımı
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label="",
                        key=unique_slider_key
                    )
                    
                    # Slider değerini kaydet
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
    
    elif param_tabs == '🎯 Temel Parametreler (V1-V7)':
        st.markdown("### 📊 Temel Risk Faktörleri")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="basic_params_1"):
                ui.element("h4", children=["🎯 Grup A (V1-V4)"], className="text-blue-600 font-semibold mb-3", key="group_a_title")
                for i in range(4):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_basic_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - Risk Faktörü {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
                    
        with param_cols[1]:
            with ui.card(key="basic_params_2"):
                ui.element("h4", children=["⚡ Grup B (V5-V7)"], className="text-green-600 font-semibold mb-3", key="group_b_title")
                for i in range(4, 7):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_basic_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - Davranış Skoru {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
    
    elif param_tabs == '⚡ Gelişmiş (V8-V14)':
        st.markdown("### ⚡ Gelişmiş Analiz Parametreleri")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="advanced_params_1"):
                ui.element("h4", children=["📈 İşlem Paterni (V8-V11)"], className="text-purple-600 font-semibold mb-3", key="pattern_title")
                for i in range(7, 11):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_advanced_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - İşlem Paterni {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
                    
        with param_cols[1]:
            with ui.card(key="advanced_params_2"):
                ui.element("h4", children=["🔍 Anomali Tespiti (V12-V14)"], className="text-orange-600 font-semibold mb-3", key="anomaly_title")
                for i in range(11, 14):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_advanced_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - Anomali Skoru {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
    
    elif param_tabs == '🔬 Uzman (V15-V21)':
        st.markdown("### 🔬 Uzman Seviye Parametreler")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="expert_params_1"):
                ui.element("h4", children=["🛡️ Güvenlik Skorları (V15-V18)"], className="text-red-600 font-semibold mb-3", key="security_title")
                for i in range(14, 18):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_expert_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - Güvenlik Skoru {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
                    
        with param_cols[1]:
            with ui.card(key="expert_params_2"):
                ui.element("h4", children=["🎖️ Risk Profili (V19-V21)"], className="text-indigo-600 font-semibold mb-3", key="risk_profile_title")
                for i in range(18, 21):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_expert_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - Risk Profili {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
    
    elif param_tabs == '🚀 Pro (V22-V28)':
        st.markdown("### 🚀 Profesyonel AI Parametreleri")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="pro_params_1"):
                ui.element("h4", children=["🧠 Makine Öğrenmesi (V22-V25)"], className="text-teal-600 font-semibold mb-3", key="ml_title")
                for i in range(21, 25):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_pro_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - ML Faktörü {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
                    
        with param_cols[1]:
            with ui.card(key="pro_params_2"):
                ui.element("h4", children=["🎯 Doğruluk Artırıcıları (V26-V28)"], className="text-cyan-600 font-semibold mb-3", key="accuracy_title")
                for i in range(25, 28):
                    param_num = i + 1
                    unique_key = f"v_param_{param_num}_pro_tab"
                    
                    slider_result = ui.slider(
                        default_value=[0.0],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{param_num} - Doğruluk Faktörü {param_num}",
                        key=unique_key
                    )
                    
                    if slider_result is not None and len(slider_result) > 0:
                        v_features[f'V{param_num}'] = slider_result[0]
                    else:
                        v_features[f'V{param_num}'] = 0.0
    
    # Parametre özeti
    st.markdown("---")
    summary_cols = st.columns([2, 1])
    
    with summary_cols[0]:
        # Progress bar for non-zero parameters
        param_count = sum(1 for key, value in v_features.items() if (isinstance(value, (int, float)) and value != 0.0) or (isinstance(value, list) and len(value) > 0 and value[0] != 0.0))
        progress_value = param_count / 28
        st.markdown(f"**📊 Parametre Kullanımı: {param_count}/28**")
        st.progress(progress_value)
        if param_count > 0:
            st.success(f"✅ {param_count} parametre aktif olarak ayarlandı")
        else:
            st.info("ℹ️ Tüm parametreler varsayılan değerlerde (0.0)")
    
    with summary_cols[1]:
        ui.metric_card(
            title="Ayarlanan Parametreler",
            content=str(param_count),
            description=f"{28} parametreden",
            key="param_count_metric"
        )
    
    # Eksik parametreleri doldur - tüm V1-V28 parametrelerinin mevcut olduğundan emin ol
    for i in range(1, 29):
        if f'V{i}' not in v_features:
            v_features[f'V{i}'] = [0.0]
    
    # Analiz sonuçları
    if analyze_btn:
        try:
            st.markdown('<div class="section-header">🔄 Analiz İşleniyor...</div>', unsafe_allow_html=True)
            
            # Input değerlerini güvenli şekilde hazırla
            try:
                amount_val = float(amount) if amount and str(amount).strip() else 100.0
            except (ValueError, TypeError):
                amount_val = 100.0
                st.warning("⚠️ Geçersiz miktar değeri, varsayılan değer (100.0) kullanılıyor.")
            
            try:
                time_val = float(time_seconds) if time_seconds and str(time_seconds).strip() else 3600.0
            except (ValueError, TypeError):
                time_val = 3600.0
                st.warning("⚠️ Geçersiz zaman değeri, varsayılan değer (3600) kullanılıyor.")
            
            # V features'ları güvenli şekilde al
            v_feature_values = []
            for i in range(1, 29):
                # v_features artık liste değil, doğrudan sayı değeri tutuyor
                feature_value = v_features.get(f'V{i}', 0.0)
                if isinstance(feature_value, list):
                    v_feature_values.append(feature_value[0] if len(feature_value) > 0 else 0.0)
                else:
                    v_feature_values.append(float(feature_value) if feature_value is not None else 0.0)
            
            # Model ve scaler kontrolü
            if model is None or scaler is None:
                st.error("❌ Model veya scaler yüklenemedi! Lütfen sayfayı yenileyin.")
                return
            
            # Sadece Time ve Amount'ı scale et
            time_amount_scaled = scaler.transform([[time_val, amount_val]])
            time_scaled = time_amount_scaled[0][0]
            amount_scaled = time_amount_scaled[0][1]
            
            # Doğru sıralama: V1-V28, Time, Amount
            features_final = v_feature_values + [time_scaled, amount_scaled]
            features_array = np.array(features_final).reshape(1, -1)
            
            # Model prediction
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            
            # Sonuçları göster
            st.markdown('<div class="section-header">📋 Analiz Sonuçları</div>', unsafe_allow_html=True)
            
            result_cols = st.columns(3)
            
            with result_cols[0]:
                if prediction == 1:
                    ui.metric_card(
                        title="⚠️ DOLANDIRICILIK",
                        content="TESPİT EDİLDİ",
                        description="Yüksek risk seviyesi",
                        key="fraud_result"
                    )
                else:
                    ui.metric_card(
                        title="✅ NORMAL",
                        content="GÜVENLİ İŞLEM",
                        description="Düşük risk seviyesi",
                        key="normal_result"
                    )
            
            with result_cols[1]:
                risk_score = probability[1] * 100
                ui.metric_card(
                    title="📊 Risk Skoru",
                    content=f"%{risk_score:.1f}",
                    description="Dolandırıcılık olasılığı",
                    key="risk_score"
                )
            
            with result_cols[2]:
                confidence = max(probability) * 100
                ui.metric_card(
                    title="🎯 Güven Seviyesi",
                    content=f"%{confidence:.1f}",
                    description="Model güvenilirliği",
                    key="confidence"
                )
            
            # Risk analizi grafiği
            st.markdown("### 📊 Risk Seviyesi Görselleştirmesi")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Seviyesi (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # İşlem detayları özeti
            with ui.card(key="transaction_summary"):
                ui.element("h4", children=["📄 İşlem Detayları"], key="transaction_summary_title")
                ui.element("p", children=[f"💰 Miktar: ${amount_val:,.2f}"], key="summary_amount")
                ui.element("p", children=[f"⏰ Zaman: {time_val:,.0f} saniye"], key="summary_time")
                ui.element("p", children=[f"🔧 Aktif Parametreler: {param_count}/28"], key="summary_params")
                ui.element("p", children=[f"🎯 Sonuç: {'Dolandırıcılık Tespit Edildi' if prediction == 1 else 'Normal İşlem'}"], key="summary_result")
            
            st.success("✅ Analiz tamamlandı!")
            
        except Exception as e:
            st.error(f"❌ Analiz sırasında hata oluştu: {str(e)}")
            st.error("Lütfen input değerlerini kontrol edin ve tekrar deneyin.")

# Raporlar Dashboard
def reports_dashboard():
    st.markdown('<div class="section-header">📈 Detaylı Raporlar ve Analizler</div>', unsafe_allow_html=True)
    
    # Report type selection
    report_tab = ui.tabs(
        options=['📊 Günlük Rapor', '📅 Haftalık Trend', '🎯 Risk Analizi', '📋 Özet Rapor'],
        default_value='📊 Günlük Rapor',
        key="reports_navigation"
    )
    
    if report_tab == '📊 Günlük Rapor':
        # Daily report metrics
        daily_cols = st.columns(4)
        
        with daily_cols[0]:
            ui.metric_card(
                title="Bugün İşlem",
                content="12,847",
                description="+8.2% dün ile karşılaştırıldığında",
                key="daily_transactions"
            )
        
        with daily_cols[1]:
            ui.metric_card(
                title="Bugün Dolandırıcılık",
                content="23",
                description="0.18% oran",
                key="daily_fraud"
            )
        
        with daily_cols[2]:
            ui.metric_card(
                title="Engellenen İşlem",
                content="19",
                description="82.6% başarı oranı",
                key="blocked_transactions"
            )
        
        with daily_cols[3]:
            ui.metric_card(
                title="Yalancı Alarm",
                content="4",
                description="17.4% hata oranı",
                key="false_positives"
            )
        
        # Hourly transaction chart
        st.markdown("### 📊 Saatlik İşlem Dağılımı")
        hours = list(range(24))
        transactions = [random.randint(300, 800) for _ in hours]
        fraud_detected = [random.randint(0, 5) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=transactions,
            mode='lines+markers',
            name='Toplam İşlem',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=hours,
            y=fraud_detected,
            mode='lines+markers',
            name='Dolandırıcılık Tespit',
            line=dict(color='#dc2626', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Saatlik İşlem ve Dolandırıcılık Dağılımı',
            xaxis_title='Saat',
            yaxis_title='İşlem Sayısı',
            yaxis2=dict(title='Dolandırıcılık Sayısı', overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_tab == '📅 Haftalık Trend':
        st.markdown("### 📈 Haftalık Trend Analizi")
        
        # Weekly data
        days = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        this_week = [random.randint(15000, 25000) for _ in days]
        last_week = [random.randint(14000, 24000) for _ in days]
        
        # Create comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Bu Hafta',
            x=days,
            y=this_week,
            marker_color='#667eea'
        ))
        fig.add_trace(go.Bar(
            name='Geçen Hafta',
            x=days,
            y=last_week,
            marker_color='#94a3b8'
        ))
        
        fig.update_layout(
            title='Haftalık İşlem Karşılaştırması',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly summary cards
        week_cols = st.columns(3)
        
        with week_cols[0]:
            total_this_week = sum(this_week)
            total_last_week = sum(last_week)
            change = ((total_this_week - total_last_week) / total_last_week) * 100
            
            ui.metric_card(
                title="Bu Hafta Toplam",
                content=f"{total_this_week:,}",
                description=f"{change:+.1f}% değişim",
                key="weekly_total"
            )
        
        with week_cols[1]:
            avg_daily = total_this_week / 7
            ui.metric_card(
                title="Günlük Ortalama",
                content=f"{avg_daily:,.0f}",
                description="işlem",
                key="daily_average"
            )
        
        with week_cols[2]:
            peak_day = days[this_week.index(max(this_week))]
            ui.metric_card(
                title="En Yoğun Gün",
                content=peak_day,
                description=f"{max(this_week):,} işlem",
                key="peak_day"
            )
    
    elif report_tab == '🎯 Risk Analizi':
        st.markdown("### 🎯 AI Model Analiz Raporları")
        
        # Visual reports navigation
        visual_tab = ui.tabs(
            options=['📊 Veri Analizi', '🎯 Feature Importance', '📈 Model Karşılaştırma', '🔬 Clustering Analizi'],
            default_value='📊 Veri Analizi',
            key="visual_reports_tabs"
        )
        
        if visual_tab == '📊 Veri Analizi':
            st.markdown("### 📊 Kapsamlı Veri Analizi Raporu")
            st.markdown("*Veri seti üzerinde yapılan detaylı analiz sonuçları*")
            
            try:
                image_path = os.path.join(os.getcwd(), "data_analysis.png")
                if os.path.exists(image_path):
                    st.image(image_path, caption="🔍 Advanced Fraud Detection - Data Analysis", use_container_width=True)
                else:
                    st.error(f"📊 data_analysis.png dosyası bulunamadı. Aranılan konum: {image_path}")
                
                with ui.card(key="data_analysis_summary"):
                    ui.element("h4", children=["📈 Analiz Özeti"], key="data_summary_title")
                    ui.element("p", children=["• Sınıf Dağılımı: %99.8 Normal, %0.2 Dolandırıcılık"], key="data_1")
                    ui.element("p", children=["• En Önemli Feature'lar: V17, V14, V12, V10"], key="data_2")
                    ui.element("p", children=["• Zaman Dağılımı: 48 saatlik dönem analizi"], key="data_3")
                    ui.element("p", children=["• Miktar Analizi: Dolandırıcılık işlemleri daha düşük miktarlarda"], key="data_4")
                    
            except Exception as e:
                st.error(f"📊 Veri analizi görseli yüklenirken hata: {str(e)}")
        
        elif visual_tab == '🎯 Feature Importance':
            st.markdown("### 🎯 En Önemli 10 Feature Analizi")
            st.markdown("*AI modelinin karar verirken en çok önemsediği parametreler*")
            
            try:
                image_path = os.path.join(os.getcwd(), "feature_importance.png")
                if os.path.exists(image_path):
                    st.image(image_path, caption="🏆 En Önemli Feature'lar - V17 lider!", use_container_width=True)
                else:
                    st.error(f"🎯 feature_importance.png dosyası bulunamadı. Aranılan konum: {image_path}")
                
                with ui.card(key="feature_importance_summary"):
                    ui.element("h4", children=["🏆 Feature Importance Özeti"], key="feature_summary_title")
                    ui.element("p", children=["• 1. V17: %18.7 önem (En kritik feature)"], key="feature_1")
                    ui.element("p", children=["• 2. V14: %17.1 önem (İkinci en önemli)"], key="feature_2")
                    ui.element("p", children=["• 3. V12: %10.6 önem (Üçüncü sırada)"], key="feature_3")
                    ui.element("p", children=["• Top 10 feature toplam etkisi: %85+"], key="feature_4")
                    
            except Exception as e:
                st.error(f"🎯 Feature importance görseli yüklenirken hata: {str(e)}")
        
        elif visual_tab == '📈 Model Karşılaştırma':
            st.markdown("### 📈 Kapsamlı Model Performans Karşılaştırması")
            st.markdown("*12 farklı AI modelinin detaylı performans analizi*")
            
            try:
                image_path = os.path.join(os.getcwd(), "model_results.png")
                if os.path.exists(image_path):
                    st.image(image_path, caption="🤖 Model Karşılaştırma - Random Forest Kazanan!", use_container_width=True)
                else:
                    st.error(f"📈 model_results.png dosyası bulunamadı. Aranılan konum: {image_path}")
                
                # Model performance summary
                model_cols = st.columns(3)
                
                with model_cols[0]:
                    ui.metric_card(
                        title="En İyi Model",
                        content="Random Forest",
                        description="F1-Score: 0.874",
                        key="best_model_card"
                    )
                
                with model_cols[1]:
                    ui.metric_card(
                        title="En Hızlı Model",
                        content="Linear SVM",
                        description="Optimize edilmiş",
                        key="fastest_model_card"
                    )
                
                with model_cols[2]:
                    ui.metric_card(
                        title="En Yüksek Precision",
                        content="Random Forest",
                        description="94.12% doğruluk",
                        key="highest_precision_card"
                    )
                
                # Raporlar için detaylı model bilgileri
                st.markdown("### 🎯 Kapsamlı Model Analizi")
                
                model_analysis_cols = st.columns(2)
                
                with model_analysis_cols[0]:
                    st.markdown("""
                    **🏆 En İyi Performans:**
                    - **Random Forest:** %87.4 F1-Score
                    - **XGBoost:** %80.6 F1-Score  
                    - **Decision Tree:** %81.1 F1-Score
                    - **Neural Network (SMOTE):** %79.8 F1-Score
                    
                    **⚡ En Hızlı Modeller:**
                    - **Linear SVM:** Optimize edilmiş hız
                    - **Logistic Regression:** Hızlı tahmin
                    - **Naive Bayes:** Düşük kaynak kullanımı
                    """)
                
                with model_analysis_cols[1]:
                    st.markdown("""
                    **🔬 Teknik Bulgular:**
                    - **En İyi Sampling:** Original ve SMOTE
                    - **En Stabil:** Random Forest 
                    - **En Yüksek Recall:** Isolation Forest (%87.6)
                    - **En Dengeli:** Neural Network ailesi
                    
                    **📊 Genel Sonuç:**
                    - 12 model kapsamlı test edildi
                    - Random Forest açık ara kazandı
                    - Modern AI teknikleri başarılı
                    """)
                    
            except Exception as e:
                st.error(f"📈 Model karşılaştırma görseli yüklenirken hata: {str(e)}")
        
        elif visual_tab == '🔬 Clustering Analizi':
            st.markdown("### 🔬 t-SNE ve PCA Clustering Analizi")
            st.markdown("*Veri noktalarının görsel cluster analizi*")
            
            try:
                image_path = os.path.join(os.getcwd(), "clustering_analysis.png")
                if os.path.exists(image_path):
                    st.image(image_path, caption="🔬 Clustering Visualization - Normal vs Fraud Ayrımı", use_container_width=True)
                else:
                    st.error(f"🔬 clustering_analysis.png dosyası bulunamadı. Aranılan konum: {image_path}")
                
                with ui.card(key="clustering_summary"):
                    ui.element("h4", children=["🔬 Clustering Analizi Özeti"], key="clustering_summary_title")
                    ui.element("p", children=["• t-SNE: Net cluster ayrımı gözlemlendi"], key="clustering_1")
                    ui.element("p", children=["• PCA: İlk 2 bileşen %61.2 + %8.4 = %69.6 variance"], key="clustering_2")
                    ui.element("p", children=["• Dolandırıcılık pattern'ları belirgin şekilde ayrılıyor"], key="clustering_3")
                    ui.element("p", children=["• Modelin başarısının görsel kanıtı"], key="clustering_4")
                    
            except Exception as e:
                st.error(f"🔬 Clustering analizi görseli yüklenirken hata: {str(e)}")
    
    elif report_tab == '📋 Özet Rapor':
        st.markdown("### 📋 Kapsamlı Sistem Özeti")
        
        # Executive summary
        summary_cols = st.columns(2)
        
        with summary_cols[0]:
            with ui.card(key="performance_summary"):
                ui.element("h4", children=["🎯 Performans Özeti"], key="perf_summary_title")
                ui.element("p", children=["• Toplam İşlem: 2.8M (bu ay)"], key="summary_1")
                ui.element("p", children=["• Dolandırıcılık Tespiti: %99.7"], key="summary_2")
                ui.element("p", children=["• Yalancı Alarm Oranı: %0.3"], key="summary_3")
                ui.element("p", children=["• Sistem Çalışma Süresi: %99.99"], key="summary_4")
                ui.element("p", children=["• Ortalama Yanıt Süresi: 45ms"], key="summary_5")
        
        with summary_cols[1]:
            with ui.card(key="financial_impact"):
                ui.element("h4", children=["💰 Mali Etki"], key="financial_title")
                ui.element("p", children=["• Engellenen Zarar: $2.4M"], key="financial_1")
                ui.element("p", children=["• Sistem Maliyeti: $45K"], key="financial_2")
                ui.element("p", children=["• Net Kazanç: $2.35M"], key="financial_3")
                ui.element("p", children=["• ROI: %5,200"], key="financial_4")
                ui.element("p", children=["• Aylık Tasarruf: $780K"], key="financial_5")
        
        # Download report button
        st.markdown("### 📥 Rapor İndirme")
        download_cols = st.columns(3)
        
        with download_cols[0]:
            if ui.button("📊 PDF Raporu İndir", key="download_pdf"):
                st.success("PDF raporu hazırlanıyor...")
        
        with download_cols[1]:
            if ui.button("📈 Excel Raporu İndir", key="download_excel"):
                st.success("Excel raporu hazırlanıyor...")
        
        with download_cols[2]:
            if ui.button("📧 Email Gönder", key="send_email"):
                st.success("Rapor email ile gönderiliyor...")

# Sayfa navigasyonu
def main():
    # Enhanced Modern Sidebar
    with st.sidebar:
        # Header logo ve title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
             border-radius: 15px; margin-bottom: 1.5rem; color: white;">
            <h1 style="margin: 0; font-size: 1.5rem;">🛡️ AI Güvenlik</h1>
            <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">Premium Edition</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu with icons
        page = option_menu(
            menu_title=None,
            options=["Ana Panel", "Dolandırıcılık Tespiti", "Raporlar"],
            icons=["speedometer2", "search", "bar-chart-line"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#667eea", "font-size": "18px"}, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "2px",
                    "padding": "12px",
                    "border-radius": "10px",
                    "--hover-color": "#f0f2f6"
                },
                "nav-link-selected": {
                    "background-color": "#667eea",
                    "color": "white",
                    "font-weight": "500"
                },
            }
        )
        
        st.markdown("---")
        
        # System Status Card
        with st.container():
            st.markdown("### 📊 Sistem Durumu")
            
            # Real-time status indicators
            status_data = {
                "Sistem": "🟢 AKTİF",
                "Model": "🟢 ÇALIŞIYOR", 
                "API": "🟢 ERİŞİLEBİLİR",
                "Veritabanı": "🟢 BAĞLI"
            }
            
            for key, value in status_data.items():
                st.markdown(f"**{key}:** {value}")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### ⚡ Hızlı İstatistikler")
        
        # Mini metrics
        quick_stats = {
            "Bugün İşlem": f"{random.randint(12000, 15000):,}",
            "Tespit Edilen": f"{random.randint(18, 35)}",
            "Başarı Oranı": f"{random.uniform(99.1, 99.9):.1f}%",
            "Uptime": "99.99%"
        }
        
        for stat, value in quick_stats.items():
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 8px; border-radius: 8px; margin: 5px 0; border-left: 3px solid #667eea;">
                <small style="color: #64748b;">{stat}</small><br>
                <strong style="color: #1e293b;">{value}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Information
        st.markdown("### 🤖 Aktif Model")
        
        active_model_info = {
            "Model": "🌲 Rastgele Orman",
            "Doğruluk": "99.9%",
            "F1-Skoru": "87.4%",
            "Son Eğitim": "2 saat önce"
        }
        
        for key, value in active_model_info.items():
            st.markdown(f"**{key}:** {value}")
        
        # Model change button
        if st.button("🔄 Model Değiştir", use_container_width=True):
            st.info("Model değişikliği için Ana Panel'i kullanın")
        
        st.markdown("---")
        
        # Security Alerts
        st.markdown("### 🚨 Güvenlik Uyarıları")
        
        # Recent alerts (simulated)
        alerts = [
            {"type": "🟡", "msg": "Orta risk işlem", "time": "5 dk"},
            {"type": "🔴", "msg": "Yüksek risk tespit", "time": "1 sa"},
            {"type": "🟢", "msg": "Sistem güncellemesi", "time": "2 sa"}
        ]
        
        for alert in alerts:
            st.markdown(f"""
            <div style="background: #fefefe; padding: 6px; border-radius: 6px; margin: 3px 0; 
                 border: 1px solid #e2e8f0; font-size: 0.85rem;">
                {alert['type']} {alert['msg']}<br>
                <small style="color: #64748b;">{alert['time']} önce</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer info
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 10px; margin-top: 1rem;">
            <small style="color: #64748b;">
                <strong>AI Güvenlik Merkezi</strong><br>
                Premium Edition v2.1<br>
                © 2024 - Tüm hakları saklıdır
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    # Sayfa içeriği
    if page == 'Ana Panel':
        main_dashboard()
    elif page == 'Dolandırıcılık Tespiti':
        fraud_detection_page()
    elif page == 'Raporlar':
        reports_dashboard()

if __name__ == "__main__":
    main() 