# %% [markdown]
# # Aprendizaje No Supervisado 2025
# 
# ---
# 
# ## 1. Análisis exploratorio inicial de la base

# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1000)
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
from plotly.offline import init_notebook_mode, plot,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.figure_factory as ff
import seaborn as sns
from sklearn.cluster import KMeans , MeanShift
from sklearn import decomposition
from sklearn.preprocessing import MultiLabelBinarizer

# %%
df=pd.read_csv('./players.csv')
df.head(5)

# %%
print(f"Shape del dataset: {df.shape}\n")

print(f"Variables: {df.columns}")

# %%
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n")

print(df.duplicated().sum())

print("\n")

print(df.describe())

# %% [markdown]
# ---
# 
# ## 2. Definición de las variables
# 
# ### Metadatos / índices
# 
# - **Unnamed: 0.1 / Unnamed: 0**: índices automáticos creados al exportar el dataset 🗑️
# - **Rank**: Ranking o posición en la lista. 🗑️
# - **Name**: Nombre del jugador. 
# - **url**: Link al perfil del jugador en la fuente original. 🗑️
# 
# ### Atributos generales
# 
# - **OVR**: Overall rating (valoración global del jugador). 
# - **PAC**: Pace (ritmo: velocidad y aceleración).
# - **SHO**: Shooting (tiro).
# - **PAS**: Passing (pases).
# - **DRI**: Dribbling (regate).
# - **DEF**: Defense (defensa).
# - **PHY**: Physical (físico).
# 
# ### Velocidad
# 
# - **Acceleration**: Aceleración.
# - **Sprint Speed**: Velocidad máxima.
# 
# ### Tiro
# 
# - **Positioning**: Posicionamiento ofensivo.
# - **Finishing**: Definición.
# - **Shot Power**: Potencia de tiro.
# - **Long Shots**: Tiros lejanos.
# - **Volleys**: Voleas.
# - **Penalties**: Penales.
# 
# ### Pases
# 
# - **Vision**: Visión de juego.
# - **Crossing**: Centros.
# - **Free Kick Accuracy**: Precisión de tiros libres.
# - **Short Passing**: Pase corto.
# - **Long Passing**: Pase largo.
# - **Curve**: Efecto.
# 
# ### Regate
# 
# - **Dribbling**: Habilidad de driblar.
# - **Agility**: Agilidad.
# - **Balance**: Balance.
# - **Reactions**: Reacciones.
# - **Ball Control**: Control de balón.
# - **Composure**: Compostura (mantener la calma bajo presión).
# 
# ### Defensa 
# 
# - **Interceptions**: Intercepciones.
# - **Heading Accuracy**: Precisión de cabezazo.
# - **Def Awareness**: Colocación defensiva.
# - **Standing Tackle**: Entrada de pie.
# - **Sliding Tackle**: Entrada al suelo.
# 
# ### Físico 
# 
# - **Jumping**: Salto.
# - **Stamina**: Resistencia.
# - **Strength**: Fuerza.
# - **Aggression**: Agresividad.
# 
# ### Información personal 
# 
# - **Position**: Posición principal (ej: ST, CAM, RB).
# - **Alternative positions**: Posiciones alternativas.
# - **Weak foot**: Puntuación del pie malo (1–5). 🗑️
# - **Skill moves**: Número de estrellas de skills (1–5). 🗑️
# - **Preferred foot**: Pie preferido (left/right). 🗑️
# - **Height**: Altura. 🗑️
# - **Weight**: Peso. 🗑️
# - **Age**: Edad. 🗑️
# - **Nation**: Nacionalidad. 🗑️ 
# - **League**: Liga en la que juega. 🗑️
# - **Team**: Equipo actual. 🗑️
# - **Play style**: Estilo de juego (solo la mitad de los jugadores tiene esta información). 🗑️
# 
# ### Estadísticas de arquero 
# 
# - **GK Diving**: Estirada.
# - **GK Handling**: Blocaje.
# - **GK Kicking**: Saque con el pie.
# - **GK Positioning**: Posicionamiento.
# - **GK Reflexes**: Reflejos.
# 
# ***
# 
# ## Criterios a definir y suposiciones
# 
# 1. Para que un jugador sea reemplazable por otro debería estar en la misma posición o en una posición alternativa (`Alternative positions`).
# 2. No todos los jugadores con la misma posición son buenos reemplazos, ya que pueden variar en habilidades, estadísticas y formas de jugar.

# %%
# Elimino las columnas que no parecen relevantes
players = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "Rank", "url", "Weak foot", "Skill moves", "Preferred foot", "Height", "Weight", "Age", "Nation", "League", "play style"])
# Pongo "" si la Position es nan
players["Position"] = players["Position"].fillna("")
players["Alternative positions"] = players["Alternative positions"].fillna("")
players["Alternative positions"] = players["Alternative positions"].apply(lambda x: [pos.strip() for pos in x.split(",") if pos != ""])
# Eliminar NaN de GK
players[["GK Reflexes", "GK Diving", "GK Handling", "GK Kicking", "GK Positioning"]] = players[["GK Reflexes", "GK Diving", "GK Handling", "GK Kicking", "GK Positioning"]].fillna(0)


# %%
# Mejor jugador por posición

players['OVR'].hist(bins = 46)

# %%
best_players_position = players.loc[players.groupby('Position')['OVR'].idxmax(), ['Position', 'Name', 'OVR']]
best_players_position

pd.DataFrame(players.Position.value_counts().sort_index())

# %%
avg_ovr = players.groupby('Team')['OVR'].mean().reset_index().sort_values("OVR", ascending=False)
avg_ovr.head(10)

# %%
n=10000 #cantidad de jugadores a considerar

df_n=players.loc[:n] #se reduce la base a los n primeros jugadores

df_n=df_n[(df_n['OVR']>70)]

skills = [
    'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY',
    'Acceleration', 'Sprint Speed', 'Positioning', 'Finishing', 'Shot Power',
    'Long Shots', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free Kick Accuracy',
    'Short Passing', 'Long Passing', 'Curve', 'Dribbling', 'Agility', 'Balance',
    'Reactions', 'Ball Control', 'Composure', 'Interceptions', 'Heading Accuracy',
    'Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Jumping', 'Stamina',
    'Strength', 'Aggression', 'GK Diving', 'GK Handling', 'GK Kicking', 'GK Positioning',
    'GK Reflexes'
]
print(len(skills), 'variables numéricas de desempeño según habilidad')

# %%
from sklearn.preprocessing import MultiLabelBinarizer

strikers = ['ST', 'CF', 'LS', 'RS']
wingers = ['LW', 'RW', 'LF', 'RF', 'LM', 'RM'] 
central_mid = ['CM', 'LCM', 'RCM']
attacking_mid = ['CAM', 'LAM', 'RAM']
defensive_mid = ['CDM', 'LDM', 'RDM']
center_backs = ['CB', 'RCB', 'LCB']
full_backs = ['LB', 'RB', 'LWB', 'RWB']
goalkeepers = ['GK']

def pos_category(position):
    if position in strikers:
        return 'Striker'
    elif position in wingers:
        return 'Winger'
    elif position in attacking_mid:
        return 'Attacking Mid'
    elif position in central_mid:
        return 'Central Mid'
    elif position in defensive_mid:
        return 'Defensive Mid'
    elif position in center_backs:
        return 'Center Back'
    elif position in full_backs:
        return 'Full Back'
    elif position in goalkeepers:
        return 'Goalkeeper'
    else:
        return 'Other'

df_n["Position2"] = df_n["Position"].apply(pos_category)

df_n['AllPositions'] = df_n.apply(lambda row: [row['Position']] + row['Alternative positions'], axis=1)

mlb = MultiLabelBinarizer()
positions_encoded = pd.DataFrame(mlb.fit_transform(df_n['AllPositions']),
                                 columns=mlb.classes_,
                                 index=df_n.index)

df_encoded = pd.concat([df_n, positions_encoded], axis=1)
df_skills = df_encoded[skills + ['Position2']]
sns.pairplot(df_skills[skills[0:10] + ['Position2']], hue='Position2', palette='bright', diag_kind='kde', plot_kws={'alpha':0.6, 's':30})
plt.suptitle('Pairplot de skills (primeras 10) por categoría de posición', y=1.02)
plt.show()

# %% [markdown]
# ---
# ## 4. Aplicación de clustering

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Escalar solo las habilidades (skills)
scaler = StandardScaler()
skills_scaled = scaler.fit_transform(df_encoded[skills])

# Convertir de nuevo a DataFrame con los mismos índices
skills_scaled_df = pd.DataFrame(skills_scaled, columns=skills, index=df_encoded.index)

# Concatenar habilidades escaladas con posiciones one-hot
X = pd.concat([skills_scaled_df, df_encoded[mlb.classes_]], axis=1)

range_n_clusters = [2, 3, 4, 5, 6]
sse = {}

skill_1 = skills[0]
skill_2 = skills[1]

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
    cluster_labels = clusterer.fit_predict(X)
    sse[n_clusters] = clusterer.inertia_

    # Calcular silueta promedio
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"Para n_clusters = {n_clusters}, el silhouette_score promedio es: {silhouette_avg:.3f}")

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("Gráfico de silueta")
    ax1.set_xlabel("Coeficiente de silueta")
    ax1.set_ylabel("Muestras")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Gráfico de clusters en dos dimensiones
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[skill_1], X[skill_2], marker='.', s=30, lw=0,
                alpha=0.7, c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker=f'${i}$', alpha=1, s=50, edgecolor='k')

    ax2.set_title("Clusters visualizados")
    ax2.set_xlabel(skill_1)
    ax2.set_ylabel(skill_2)

    plt.suptitle((f"Análisis de silueta para KMeans con n_clusters = {n_clusters}"),
                 fontsize=14, fontweight='bold')
    plt.show()

# Método del codo
plt.figure(figsize=(8,5))
plt.plot(list(sse.keys()), list(sse.values()), 'o-')
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del codo para KMeans")
plt.show()

# %%
kmeans_final = KMeans(n_clusters=4, random_state=10, n_init=10)
df_n["Cluster"] = kmeans_final.fit_predict(X)

ct = pd.crosstab(df_n["Cluster"], df_n["Position2"], normalize=True)

# Visualización
ct.plot(kind="bar", stacked=True, figsize=(12,6), colormap="tab20")
plt.title("Distribución de posiciones por cluster")
plt.ylabel("Proporción dentro del cluster")
plt.xlabel("Cluster")
plt.legend(title="Position2", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# 
# ## Pregunta: ¿Se realizó alguna normalización o escalado de la base? ¿Por qué ?
# 
# Se utilizó StandardScaler para escalar todas las variables numéricas (skills) y luego concatenar las posiciones codificadas.
# 
# De esta forma:
# 
# - Todas las habilidades tendrán media 0 y desviación estándar 1.
# - Las variables categóricas (posiciones one-hot, ya en 0/1) no quedan desbalanceadas.

# %% [markdown]
# ---
# # Conclusiones
# 
# ### ¿Qué hay en cada cluster?
# 
# En cada cluster se agrupan jugadores según su especialización posicional (arqueros, defensores centrales, delanteros).
# 
# - Cluster 0: Este cluster agrupa perfiles polivalentes, jugadores de medio campo y laterales con roles mixtos.
# 
# - Cluster 1: Este cluster representa a los jugadores defensivos especialistas.
# 
# - Cluster 2: Los arqueros están perfectamente aislados.
# 
# - Cluster 3: Principalmente Wingers (Extremos) y Strikers (Delanteros). Este cluster representa a los jugadores ofensivos y de banda.
# 
# ### ¿Son efectivamente equivalentes los jugadores de un cluster, es decir, podrían cumplir el mismo rol en un equipo?
# 
# No son realmente equivalentes: el clustering se hace en un espacio multidimensional y la visualización en 2D (PAC y SHO) oculta diferencias importantes. Además, los perfiles de habilidades dentro de un mismo cluster son heterogéneos y el agrupamiento no considera roles tácticos.
# 
# ### Si se trata de clusters heterogéneos, ¿por qué razón pueden haber sido agrupadas las jugadoras del cluster?
# 
# - Pueden tener perfiles globales similares aunque difieran en habilidades específicas.
# - La similitud surge en otras dimensiones (pase, defensa, regate, etc.).
# - El algoritmo detecta jugadores de nivel global parecido.
# 
# ### ¿Qué motiva las diferencias en tamaño?
# - Algunos arquetipos de jugador son más comunes que otros.
# - Reflejan la distribución real de jugadores en el dataset.
# - Hay combinaciones de habilidades más frecuentes (ej. extremos rápidos).
# - Los clusters grandes suelen corresponder a jugadores promedio, y los pequeños a élites o especialistas raros.


