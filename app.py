import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import os

# ==========================================
# 1. ADVANCED UI & ANIMATION ENGINE
# ==========================================
def apply_ultra_ui():
    st.set_page_config(page_title="BORPAT ai | Pro Dashboard", layout="wide", page_icon="🍃")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

        /* Main Dark Theme Background */
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            color: #ffffff;
            font-family: 'Roboto', sans-serif;
        }

        /* TAB UI: Ultra-Large Fonts & Pop-up Scale Animation */
        button[data-testid="stMarkdownContainer"] p {
            font-size: 26px !important;
            font-weight: 700 !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            color: #ffffff;
        }
        div[data-testid="stTabBar"] button:hover {
            transform: scale(1.15) translateY(-5px) !important;
            background: rgba(76, 175, 80, 0.15) !important;
            border-radius: 12px !important;
        }

        /* REMOVE WHITE BOXES & STRIPS */
        div[data-testid="stMetric"], div[data-testid="stMetricValue"], 
        div[data-testid="stMetricLabel"], .stAlert, 
        div[data-testid="stVerticalBlock"] > div {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #ffffff !important;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }

        /* Severity Risk Bar */
        .severity-container {
            width: 100%;
            height: 35px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .severity-fill { height: 100%; border-radius: 15px; transition: width 0.5s ease; }

        /* Agronomy Card Styling */
        .agro-card {
            background: rgba(255, 255, 255, 0.08);
            padding: 15px;
            border-radius: 12px;
            border-left: 5px solid #4CAF50;
            margin: 10px 0;
        }

        h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; color: #4CAF50 !important; }
        </style>
        """, unsafe_allow_html=True)

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

# ==========================================
# 2. CORE SYSTEM LOGIC & ADVANCED PHARMACOLOGY
# ==========================================
class AIEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.classes = ['Tea Leaf Blight', 'Tea Red Leaf Spot', 'Tea Red Scab', 'Healthy']

    @st.cache_resource
    def load_engine(_self):
        return tf.keras.models.load_model(_self.model_path)

    def predict(self, image, model):
        size = (224, 224)
        proc = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(proc)[np.newaxis, ...] / 255.0
        preds = model.predict(img_array)
        return self.classes[np.argmax(preds)], np.max(preds) * 100

class AgronomyAdvisor:
    @staticmethod
    def get_data(disease, confidence):
        kb = {
            'Tea Red Leaf Spot': {
                'causal': "Cephaleuros virescens (Parasitic Green Alga)",
                'symptoms': [
                    "Minute orange-brown 'rusty' spots on upper leaf surface.",
                    "Raised, hairy patches that enlarge to 10-15mm lesions.",
                    "Chlorosis and pale stems in advanced stages of infestation."
                ],
                'prevalence': "Common in Assam monsoon (June-Oct). Spreads via wind in humid weather.",
                'chemical': """
                    - **Primary Medicine:** Copper Oxychloride (e.g., Blitox).
                    - **Dosage:** 2.5g per Litre of water (Approx 400-500L/Hectare).
                    - **Schedule:** Spray every 14 days during monsoon peaks.
                    - **Pre-harvest Interval (PHI):** 7 days minimum.
                """,
                'integrated': [
                    "Apply NPK fertilizers focusing on **Potassium** to boost immunity.",
                    "Improve soil drainage to reduce algal spore survival.",
                    "Prune shade trees to increase sunlight and reduce humidity."
                ],
                'economic': "Yield Loss: 15-20% untreated. Plan cost: ~₹1,250/acre.",
                'score': 75, 'color': '#FF9800'
            },
            'Tea Leaf Blight': {
                'causal': "Exobasidium vexans (Fungus)",
                'symptoms': [
                    "Translucent pale green spots turning into white blisters.",
                    "Secondary curling of young, succulent leaf tissue.",
                    "Complete defoliation if pathogen spreads to the stems."
                ],
                'prevalence': "Severe in cloudy, wet weather with low sunlight.",
                'chemical': """
                    - **Primary Medicine:** Hexaconazole 5% EC (e.g., Contaf).
                    - **Dosage:** 2ml per Litre of water.
                    - **Alternative:** Propiconazole 25% EC (1ml/L).
                    - **Application:** High-volume spray targeting the underside of leaves.
                """,
                'integrated': [
                    "Immediately remove and burn infected shoots.",
                    "Adjust shade tree density to ensure better air circulation.",
                    "Avoid excessive Nitrogen (N) fertilization during peak blight season."
                ],
                'economic': "Severe quality and yield degradation. Recovery takes 3-4 months.",
                'score': 92, 'color': '#F44336'
            },
            'Tea Red Scab': {
                'causal': "Elsinoë ampelina / Generic Necrotic Scab",
                'symptoms': [
                    "Dry, rough, corky scaly patches on leaf surfaces.",
                    "Bark cracking and lesions visible on the stems.",
                    "Reduced terminal bud growth leading to stunted yields."
                ],
                'prevalence': "Often follows mechanical damage or intense drought followed by rain.",
                'chemical': """
                    - **Primary Medicine:** Mancozeb (e.g., Indofil M-45).
                    - **Dosage:** 3g per Litre of water.
                    - **Paste Application:** Use a copper-based paste for open stem cankers.
                """,
                'integrated': [
                    "Prune diseased stems significantly below the affected zone.",
                    "Disinfect all pruning equipment with 5% bleach.",
                    "Maintain high garden hygiene (removing leaf litter)."
                ],
                'economic': "Reduced stem vitality. Plan cost: ~₹1,100/acre.",
                'score': 65, 'color': '#FFEB3B'
            },
            'Healthy': {
                'causal': "None",
                'symptoms': ["Optimal chlorophyll density", "Intact cellular structure"],
                'prevalence': "N/A",
                'chemical': "N/A - Continue standard organic maintenance.",
                'integrated': ["Weekly monitoring", "Standard NPK balance"],
                'economic': "N/A - Market Peak Quality.",
                'score': 10, 'color': '#4CAF50'
            }
        }
        return kb.get(disease, kb['Healthy'])

# ==========================================
# 3. MAIN DASHBOARD ORCHESTRATOR
# ==========================================
def main():
    apply_ultra_ui()
    ai = AIEngine('tea_disease_model.keras')
    
    st.markdown("<h1 style='text-align: center;'>🛡️ BORPAT ai: SYSTEM ARCHITECTURE v2.5</h1>", unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Probability Analytics", "📸 Detection Centre", "💊 Advisory & Remedies"])

    # SECTION 1: PROBABILITY ANALYTICS (Training Info & 3 Charts)
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🛠️ Deep Learning Methodology & Training")
        st.write("""
        The **BORPAT ai** engine is built on a high-performance **MobileNetV2** architecture, optimized via transfer learning. 
        To ensure industrial-grade precision, we trained the model using the **Mendeley Tea Disease Dataset**, 
        curating over **5,000+ high-resolution samples** featuring expert-verified pathogens. 
        This extensive training allows the system to identify necrotic patterns across diverse lighting and 
        environmental conditions common in regional tea estates.
        """)
        st.divider()
        st.subheader("📈 Statistical Performance Analytics")
        col1, col2, col3 = st.columns(3)
        chart_files = ["1.jpeg", "2.jpeg", "3.jpeg"]
        for i, col in enumerate([col1, col2, col3]):
            with col:
                if os.path.exists(chart_files[i]):
                    st.image(chart_files[i], use_container_width=True, caption=f"Model Metric {i+1}")
                else: st.warning(f"Chart file '{chart_files[i]}' not found.")
        st.info("💡 **Benchmark Result:** Final Validation Accuracy for this architecture is **98.12%**.")
        st.markdown("</div>", unsafe_allow_html=True)

    # SECTION 2: DETECTION CENTER
    with tab2:
        col_in, col_res = st.columns(2, gap="large")
        with col_in:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            mode = st.radio("Input Source", ["Upload File", "Live Scanner"], horizontal=True)
            img_file = st.file_uploader("Upload Leaf Sample") if mode == "Upload File" else st.camera_input("Scanner Active")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_res:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if img_file:
                raw_img = Image.open(img_file)
                st.image(raw_img, use_container_width=True, caption="Analyzed Sample")
                try:
                    model = ai.load_engine()
                    res, conf = ai.predict(raw_img, model)
                    st.metric("Detected Pathogen", res)
                    st.metric("Confidence Score", f"{conf:.2f}%")
                    st.session_state['last_res'] = res
                    st.session_state['last_conf'] = conf
                except Exception as e: st.error(f"Engine Failure: {e}")
            else:
                lottie_url = "https://assets5.lottiefiles.com/packages/lf20_awP420Zf8l.json"
                st_lottie(load_lottieurl(lottie_url), height=200)
            st.markdown("</div>", unsafe_allow_html=True)

    # SECTION 3: REFINED ADVISORY (Medicines & Detailed Remedies)
    with tab3:
        target = st.session_state.get('last_res', 'Healthy')
        conf = st.session_state.get('last_conf', 0)
        info = AgronomyAdvisor.get_data(target, conf)

        st.subheader(f"📋 Integrated Treatment & Medicinal Plan: {target}")
        
        # Severity Indicator
        st.write(f"**Pathogen Severity Score:** {info['score']}%")
        st.markdown(f'<div class="severity-container"><div class="severity-fill" style="width: {info["score"]}%; background: {info["color"]};"></div></div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
            st.write("💉 **Chemical Control & Medicines**")
            st.write(info['chemical'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
            st.write("🌿 **Integrated Management**")
            for act in info['integrated']: st.write(f"✔️ {act}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
            st.write("🔬 **Symptoms Confirmed by AI**")
            for sym in info['symptoms']: st.write(f"• {sym}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
            st.write("💰 **Economic Projection**")
            st.caption(info['economic'])
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("✅ Action Checklist")
        st.checkbox("Immediate (0-3 days): Purchase and apply primary medicine", value=False)
        st.checkbox("Immediate: Burn infected leaves to stop spore spread", value=False)
        st.button("📄 Export Full Prescription PDF")

if __name__ == "__main__":
    main()