import pandas as pd
from pycaret.classification import *
import mlflow
import shap
import matplotlib.pyplot as plt
import os

# ===============================
# Cargar dataset
# ===============================
df = pd.read_csv("fintech_credit_approval.csv")

# Normalizar columnas a minúsculas (opcional y robusto)
df.columns = df.columns.str.strip().str.lower()

# Preparar dataset
df_model = df.drop(columns=["user_id"])

# ===============================
# Configurar MLflow
# ===============================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Aprobacion_credito")

# ===============================
# PyCaret setup
# ===============================
s = setup(
    data=df_model,
    target="approved",  # Asegúrate de que exista como 'risk_level' en minúscula
    session_id=404,
    log_experiment=True,
    experiment_name="Aprobacion_credito",
    verbose=True,
    profile=False,
    use_gpu=False
)

# ===============================
# Entrenamiento y evaluación
# ===============================
best_model = compare_models()
evaluate_model(best_model)

# ===============================
# Guardar el modelo localmente
# ===============================
save_model(best_model, "credit_approval_model")
print("✅ Modelo entrenado, evaluado y guardado.")

# ===============================
# Interpretabilidad con SHAP
# ===============================
print("🧠 Generando interpretabilidad con SHAP...")

# Obtener datos transformados por PyCaret (todo numérico)
X_sample = get_config('X_train_transformed').sample(100, random_state=42)

# Crear carpeta si no existe
os.makedirs("shap_plots", exist_ok=True)

# Usamos KernelExplainer para compatibilidad con cualquier modelo
background = X_sample.median().values.reshape(1, -1)

# Función de predicción que SHAP puede usar
predict_fn = lambda x: best_model.predict_proba(pd.DataFrame(x, columns=X_sample.columns))

# Crear el explainer
explainer = shap.KernelExplainer(predict_fn, background)

# Calcular SHAP values (puede tomar unos segundos)
shap_values = explainer.shap_values(X_sample, nsamples=100)

# Graficar summary plot para la clase 1 (riesgo alto)
summary_plot_path = "shap_plots/summary_plot.png"
plt.figure()
shap.summary_plot(shap_values[1], X_sample, show=False)
plt.savefig(summary_plot_path, bbox_inches='tight')
plt.close()

# ===============================
# Registrar en MLflow
# ===============================
with mlflow.start_run(run_name="Interpretabilidad_SHAP", nested=True):
    mlflow.log_artifact(summary_plot_path)
    mlflow.log_param("modelo", str(best_model))

print("📊 Interpretabilidad SHAP registrada en MLflow.")