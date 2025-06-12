import streamlit as st
import requests

# Configuración de página
st.set_page_config(page_title="Predicción de Upselling", layout="centered")
st.title("📈 Predicción de Aceptación de Upselling")
st.markdown("Simula el comportamiento de un cliente ante una oferta adicional de seguro.")

# Inputs del cliente
age = st.slider("🎂 Edad", 25, 80, 40)
monthly_income_usd = st.number_input("💵 Ingreso mensual (USD)", min_value=0.0, step=1000.0, value=15000.0)
app_usage_score = st.number_input("📱 Escore interno uso", min_value=0.0, max_value=10.0, step=1.0, value=5.0)
digital_profile_strength = st.number_input("💾 Escore perfil digital", min_value=0.0, max_value=100.0, step=10.0, value=50.0)
num_contacts_uploaded = st.slider("📇 Número de contactos", 0, 20, 5)
residence_risk_zone = st.selectbox("💼 Riesgo zona", ["Media", "Baja", "Alta"])
political_event_last_month = st.radio("📬 Evento político", ["No", "Sí"])

# Threshold slider
threshold = st.number_input("🎯 Umbral de decisión (Threshold)", min_value=0.0, max_value=1.0, step=0.1, value=0.5)

# Botón de predicción
if st.button("🔍 Evaluar Probabilidad"):
    with st.spinner("Consultando modelo..."):
        try:
            payload = {
                "age": age,
                "monthly_income_usd": monthly_income_usd,
                "app_usage_score": app_usage_score,
                "digital_profile_strength": digital_profile_strength,
                "num_contacts_uploaded": num_contacts_uploaded,
                "residence_risk_zone": residence_risk_zone,
                "political_event_last_month": 1 if political_event_last_month == "Sí" else 0,
                "threshold": threshold
            }

            r = requests.post("http://localhost:8000/predict_aprobacion", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                score = resultado["score_probabilidad"]
                aceptara = resultado["aceptará"]

                st.markdown(f"### 🔢 Score de aceptación: **{score:.3f}**")
                st.markdown(f"### 🎯 Umbral usado: **{threshold:.2f}**")

                if aceptara:
                    st.success("✅ El cliente probablemente **aceptará** el upselling.")
                else:
                    st.warning("⚠️ El cliente probablemente **rechazará** la oferta.")
            else:
                st.error("❌ Error en la respuesta del modelo.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")
