import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(BASE_DIR, 'Social.csv') 

if not os.path.exists(file_path): # se il file non esiste
    raise FileNotFoundError(f"File non trovato: {file_path}")

# legge il file .csv e lo salva in una tabella (DataFrame)
df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'ND', 'n/a', ''], nrows=1000000)

# info dataset
print(f"Le dimensioni del dataset sono:{df.shape}\n\n")
print(f"Le prime 5 righe sono:{df.head()}\n\n")
print("Quanti valori validi ci sono per ogni colonna:")
df.info()

colonne_nulle = df.columns[df.isna().all()]
print(f"Colonne completamente nulle: {colonne_nulle.to_list()}") 

# PULIZIA DATAFRAME
df.dropna(axis=1,how='all',inplace=True) # elimino le colonne completamente nulle 

encoder=LabelEncoder() # trasformo object in int
df['Gender']=encoder.fit_transform(df['Gender'])
df['Academic_Level']=encoder.fit_transform(df['Academic_Level'])
df['Country']=encoder.fit_transform(df['Country'])
df['Affects_Academic_Performance']=encoder.fit_transform(df['Affects_Academic_Performance'])
df['Relationship_Status']=encoder.fit_transform(df['Relationship_Status'])
df.info()

# Selezioniamo i nomi delle colonne numeriche
features_names= df.select_dtypes(include=[np.number]).columns.tolist()
features_names.remove('Student_ID')

df.dropna(axis=0,how='any',subset=features_names+['Most_Used_Platform'],inplace=True) # righe
print (f"Valori per ogni classe target: {df['Most_Used_Platform'].value_counts()}") # abbastanza bilanciate

df['Most_Used_Platform']=df['Most_Used_Platform'].replace({
    "Twitter":"Altro",
    "LinkedIn":"Altro",
    "WeChat":"Altro",
    "Snapchat":"Altro",
    "LINE":"Altro",
    "KakaoTalk":"Altro",
    "VKontakte":"Altro",
    "YouTube":"Altro",
    "WhatsApp":"Altro"
})

print (f"Nuovi valori per ogni classe target: {df['Most_Used_Platform'].value_counts()}") 

X = df[features_names] 
Y = df['Most_Used_Platform'] # target

# standardizzo dati
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)


# Numero di cluster = numero di categorie Most_Used_Platform
num_clusters= Y.nunique()
print (f"Most_Used_Platform ha {num_clusters} categorie che sono {Y.unique()}")

# clustering non supervisionato (n_clusters = num gruppi il modello deve cercare nei dati)
kmeans= KMeans(n_clusters=num_clusters, random_state=42) 
labels= kmeans.fit_predict(X_scaled) # fit_predict addestra il modello sui dati numerici (X) e assegna ogni riga a un cluster

# tabella che mostra quante righe di ogni classe Most_Used_Platformy sono in ciascun cluster
crosstab= pd.crosstab(Y, labels, colnames=['cluster'])
print(f"Crosstab: {crosstab}") 

# Per ogni cluster assegna come classe predetta il Most_Used_Platform più frequente al suo interno (majority vote)
cluster_to_Most_Used_Platform= crosstab.idxmax(axis=0).to_dict()
print("Mapping cluster --> Most_Used_Platform: ", cluster_to_Most_Used_Platform) 

# Trasforma gli ID dei cluster in etichette Most_Used_Platform
pred_Most_Used_Platform= pd.Series(labels, index=df.index).map(cluster_to_Most_Used_Platform)
is_correct= pred_Most_Used_Platform.eq(Y) # confronta ogni predizione con il valore reale → booleani True/False


summary= (
    pd.DataFrame({'Most_Used_Platform': df['Most_Used_Platform'], 'correct': is_correct})
    # raggruppa per Most_Used_Platformy conta: quante righe sono corrette (True → 1) e quante righe totali ci sono
    .groupby('Most_Used_Platform')['correct']
    .agg(correct='sum', tot='count')
)


summary['incorrect']= summary['tot'] - summary['correct'] # aggiungo colonna incorect
summary['accuracy_%']= (summary['correct'] / summary['tot'] * 100).round(2) # aggiungo colonna accuracy
print(f"Performance per ogni classe di 'Most_Used_Platform': {summary}")

overall_acc= is_correct.mean() # media dei valori is_correct
print(f"Accuracy globale (cluster --> Most_Used_Platform): {overall_acc*100:.2f}%")

pca= PCA(n_components=2)
compressed_features= pca.fit_transform(X_scaled)


base_cmap= plt.colormaps["tab10"] 
colors= base_cmap.colors[:num_clusters] 
cmap= ListedColormap(colors) 

# Usiamo Matplotlib per disegnare il grafico con i cluster
plt.figure(figsize=(8,6))
sc1= plt.scatter( # Disegna un grafico a dispersione. Ogni punto rappresenta una riga del dataset
    compressed_features[:,0], compressed_features[:,1],
    c=labels, cmap=cmap, 
    alpha=0.6, s=12 
)

clusters_id= np.unique(labels)
colors= [cmap(i) for i in range(num_clusters)] # Assegnamo ad ogni cluster un colore diverso
patches= [mpatches.Patch(color=colors[i], label=f'Cluster {cl}') for i, cl in enumerate(clusters_id)]

plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)
plt.title(f'KMeans — PCA 2D (N cluster={num_clusters})')
plt.tight_layout() # Formattazione disegno
plt.show()