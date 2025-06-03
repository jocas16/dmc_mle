from pycaret.clustering import *
import pandas as pd

# 1. Cargar el dataset
df = pd.read_csv("clientes_unsupervised.csv")

# 2. Configurar PyCaret para clustering
setup(data=df, normalize=True, session_id=123) # , silent=True

# 3. Crear modelo KMeans
model = create_model('kmeans', num_clusters=4)

# 4. Asignar clústeres
df_clustered = assign_model(model)

# 5. Guardar el modelo
save_model(model, "cluster_model_clientes")

# 6. Exportar los resultados si deseas revisar
df_clustered.to_csv("clientes_clusterizados.csv", index=False)
print("Modelo guardado y clústeres asignados.")
