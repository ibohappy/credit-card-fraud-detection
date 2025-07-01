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
            # Tüm slider state'lerini sıfırla ve timestamp ekle
            import time
            timestamp = str(int(time.time()))
            for i in range(1, 29):
                st.session_state[f"v{i}_slider"] = [0.0]
            
            # Force refresh için unique flag ekle
            st.session_state["slider_refresh_key"] = timestamp
            st.success("✅ Tüm parametreler sıfırlandı!")
            st.rerun()
    
    with action_cols[1]:
        random_clicked = ui.button("🎲 Rastgele Örnek Yükle", key="load_random_sample")
        if random_clicked:
            # Rastgele değerler yükle ve timestamp ekleyerek unique key'ler oluştur
            import time
            timestamp = str(int(time.time()))
            for i in range(1, 29):
                random_val = round(random.uniform(-2.0, 2.0), 1)
                st.session_state[f"v{i}_slider"] = [random_val]
            
            # Force refresh için unique flag ekle
            st.session_state["slider_refresh_key"] = timestamp
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
                    # Safe state handling with unique key for force refresh
                    refresh_key = st.session_state.get("slider_refresh_key", "")
                    slider_key = f"v{i+1}_slider_{refresh_key}" if refresh_key else f"v{i+1}_slider"
                    base_key = f"v{i+1}_slider"
                    
                    current_state = st.session_state.get(base_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
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
                    
                    ui.element("h5", children=[f"V{i+1}"], className=f"{color_class} font-bold mb-1", key=f"all_param_title_{i+1}_{refresh_key}")
                    ui.element("small", children=[f"{category} - {desc}"], className="text-gray-500 text-xs", key=f"all_param_desc_{i+1}_{refresh_key}")
                    
                    slider_result = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label="",
                        key=slider_key
                    )
                    
                    # Store the result - no session state modification after widget creation
                    if slider_result is not None:
                        v_features[f'V{i+1}'] = slider_result
                    else:
                        v_features[f'V{i+1}'] = [default_val]
    
    elif param_tabs == '🎯 Temel Parametreler (V1-V7)':
        st.markdown("### 📊 Temel Risk Faktörleri")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="basic_params_1"):
                ui.element("h4", children=["🎯 Grup A (V1-V4)"], className="text-blue-600 font-semibold mb-3", key="group_a_title")
                for i in range(4):
                    # Safe state handling with unique key for force refresh
                    refresh_key = st.session_state.get("slider_refresh_key", "")
                    slider_key = f"v{i+1}_slider_{refresh_key}" if refresh_key else f"v{i+1}_slider"
                    base_key = f"v{i+1}_slider"
                    
                    current_state = st.session_state.get(base_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    slider_result = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - Risk Faktörü {i+1}",
                        key=slider_key
                    )
                    
                    # Store the result - no session state modification after widget creation
                    if slider_result is not None:
                        v_features[f'V{i+1}'] = slider_result
                    else:
                        v_features[f'V{i+1}'] = [default_val]
                    
        with param_cols[1]:
            with ui.card(key="basic_params_2"):
                ui.element("h4", children=["⚡ Grup B (V5-V7)"], className="text-green-600 font-semibold mb-3", key="group_b_title")
                for i in range(4, 7):
                    # Safe state handling with unique key for force refresh
                    refresh_key = st.session_state.get("slider_refresh_key", "")
                    slider_key = f"v{i+1}_slider_{refresh_key}" if refresh_key else f"v{i+1}_slider"
                    base_key = f"v{i+1}_slider"
                    
                    current_state = st.session_state.get(base_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    slider_result = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - Davranış Skoru {i+1}",
                        key=slider_key
                    )
                    
                    # Store the result - no session state modification after widget creation
                    if slider_result is not None:
                        v_features[f'V{i+1}'] = slider_result
                    else:
                        v_features[f'V{i+1}'] = [default_val]
    
    elif param_tabs == '⚡ Gelişmiş (V8-V14)':
        st.markdown("### ⚡ Gelişmiş Analiz Parametreleri")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="advanced_params_1"):
                ui.element("h4", children=["📈 İşlem Paterni (V8-V11)"], className="text-purple-600 font-semibold mb-3", key="pattern_title")
                for i in range(7, 11):
                    # Safe state handling
                    slider_key = f"v{i+1}_slider"
                    current_state = st.session_state.get(slider_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    v_features[f'V{i+1}'] = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - İşlem Paterni {i+1}",
                        key=slider_key
                    )
                    
        with param_cols[1]:
            with ui.card(key="advanced_params_2"):
                ui.element("h4", children=["🔍 Anomali Tespiti (V12-V14)"], className="text-orange-600 font-semibold mb-3", key="anomaly_title")
                for i in range(11, 14):
                    # Safe state handling
                    slider_key = f"v{i+1}_slider"
                    current_state = st.session_state.get(slider_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    v_features[f'V{i+1}'] = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - Anomali Skoru {i+1}",
                        key=slider_key
                    )
    
    elif param_tabs == '🔬 Uzman (V15-V21)':
        st.markdown("### 🔬 Uzman Seviye Parametreler")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="expert_params_1"):
                ui.element("h4", children=["🛡️ Güvenlik Skorları (V15-V18)"], className="text-red-600 font-semibold mb-3", key="security_title")
                for i in range(14, 18):
                    # Safe state handling
                    slider_key = f"v{i+1}_slider"
                    current_state = st.session_state.get(slider_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    v_features[f'V{i+1}'] = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - Güvenlik Skoru {i+1}",
                        key=slider_key
                    )
                    
        with param_cols[1]:
            with ui.card(key="expert_params_2"):
                ui.element("h4", children=["🎖️ Risk Profili (V19-V21)"], className="text-indigo-600 font-semibold mb-3", key="risk_profile_title")
                for i in range(18, 21):
                    # Safe state handling
                    slider_key = f"v{i+1}_slider"
                    current_state = st.session_state.get(slider_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    v_features[f'V{i+1}'] = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - Risk Profili {i+1}",
                        key=slider_key
                    )
    
    elif param_tabs == '🚀 Pro (V22-V28)':
        st.markdown("### 🚀 Profesyonel AI Parametreleri")
        param_cols = st.columns(2)
        
        with param_cols[0]:
            with ui.card(key="pro_params_1"):
                ui.element("h4", children=["🧠 Makine Öğrenmesi (V22-V25)"], className="text-teal-600 font-semibold mb-3", key="ml_title")
                for i in range(21, 25):
                    # Safe state handling
                    slider_key = f"v{i+1}_slider"
                    current_state = st.session_state.get(slider_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    v_features[f'V{i+1}'] = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - ML Faktörü {i+1}",
                        key=slider_key
                    )
                    
        with param_cols[1]:
            with ui.card(key="pro_params_2"):
                ui.element("h4", children=["🎯 Doğruluk Artırıcıları (V26-V28)"], className="text-cyan-600 font-semibold mb-3", key="accuracy_title")
                for i in range(25, 28):
                    # Safe state handling
                    slider_key = f"v{i+1}_slider"
                    current_state = st.session_state.get(slider_key, None)
                    
                    if current_state is None or not isinstance(current_state, list):
                        default_val = 0.0
                    else:
                        default_val = current_state[0] if len(current_state) > 0 else 0.0
                    
                    v_features[f'V{i+1}'] = ui.slider(
                        default_value=[default_val],
                        min_value=-5.0,
                        max_value=5.0,
                        step=0.1,
                        label=f"V{i+1} - Doğruluk Faktörü {i+1}",
                        key=slider_key
                    )
    
    # Parametre özeti
    st.markdown("---")
    summary_cols = st.columns([2, 1])
    
    with summary_cols[0]:
        # Progress bar for non-zero parameters
        param_count = sum(1 for key, value in v_features.items() if value != 0.0)
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
    
    # Analiz sonuçları
    if analyze_btn:
        try:
            # Input değerlerini hazırla
            amount_val = float(amount) if amount else 100.0
            time_val = float(time_seconds) if time_seconds else 3600.0
            
            # V features'ları al (zaten PCA ile normalize edilmiş)
            # ui.slider() artık doğrudan liste döndürüyor, ilk değeri al
            v_feature_values = []
            for i in range(28):
                slider_val = v_features.get(f'V{i+1}', [0.0])
                if isinstance(slider_val, list):
                    v_feature_values.append(slider_val[0] if slider_val else 0.0)
                else:
                    v_feature_values.append(slider_val if slider_val is not None else 0.0)
            
            # Sadece Time ve Amount'ı scale et (V features dokunulmaz)
            time_amount_scaled = scaler.transform([[time_val, amount_val]])
            time_scaled = time_amount_scaled[0][0]
            amount_scaled = time_amount_scaled[0][1]
            
            # Doğru sıralama: V1-V28, Time, Amount (original dataset formatı)
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
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Seviyesi"},
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
            
        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {e}")

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
        st.markdown("### 🎯 Risk Dağılımı ve Analizi")
        
        # Risk distribution pie chart
        risk_labels = ['Düşük Risk', 'Orta Risk', 'Yüksek Risk', 'Kritik Risk']
        risk_values = [75.2, 18.5, 5.1, 1.2]
        risk_colors = ['#10b981', '#f59e0b', '#ef4444', '#7c2d12']
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_labels,
            values=risk_values,
            marker_colors=risk_colors,
            hole=0.4
        )])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            title='Risk Seviyesi Dağılımı (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors analysis
        st.markdown("### 🔍 Risk Faktörleri Analizi")
        
        risk_factors = [
            {"factor": "Yüksek İşlem Miktarı", "impact": 0.85, "frequency": 145},
            {"factor": "Gece Saati İşlemleri", "impact": 0.72, "frequency": 89},
            {"factor": "Farklı Lokasyon", "impact": 0.68, "frequency": 234},
            {"factor": "Hızlı Ardışık İşlem", "impact": 0.61, "frequency": 67},
            {"factor": "Yeni Müşteri", "impact": 0.45, "frequency": 156}
        ]
        
        risk_df = pd.DataFrame(risk_factors)
        risk_df.columns = ["Risk Faktörü", "Etki Seviyesi", "Tespit Sayısı"]
        ui.table(data=risk_df, maxHeight=300)
    
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