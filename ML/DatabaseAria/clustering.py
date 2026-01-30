# Prende dati sulla qualità dell’aria, li raggruppa automaticamente in gruppi simili (cluster) senza sapere le etichette, 
# e poi controlla quanto questi gruppi assomigliano alle vere categorie di qualità dell’aria (status).
import pandas as pd
import numpy as np # manipolazione dei dati
import os
import matplotlib.pyplot as plt # visualizzazione grafica
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap # gestione colori e legende
from sklearn.preprocessing import StandardScaler # standardizzazione dei dati
from sklearn.cluster import KMeans # clustering non supervisionato
from sklearn.decomposition import PCA # riduzione dimensionale per grafici 2D
from sklearn.utils import check_random_state

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # percorso del file
file_path = os.path.join(BASE_DIR, 'air_quality.csv') # creo percorso per il file air_quality.csv

if not os.path.exists(file_path): # se il file non esiste
    raise FileNotFoundError(f"File non trovato: {file_path}")
# raise genera un'eccezione: blocca il programma e stampa messaggio di errore

# legge il file air_quality e lo salva in una tabella (DataFrame)
df = pd.read_csv( 
    file_path,
    low_memory=False, # evita warning sui tipi di dato
    na_values=['-', 'NA', 'ND', 'n/a', ''], # considera questi valori come "mancanti" (pandas gli trasforma in NaN)
    nrows=1000000 # carica solo le prime 1.000.000 righe
)

# info dataset
print(f"Le dimensioni del dataset sono:{df.shape}\n\n")
print(f"Le prime 5 righe sono:{df.head()}\n\n")
print("Quanti valori validi ci sono per ogni colonna:")
df.info()

colonne_nulle = df.columns[df.isna().all()]
print(f"Colonne completamente nulle: {colonne_nulle.to_list()}") 


# PULISCO DATASET
df.dropna(axis=1, how='all', inplace=True) # Elimina colonne completamente vuote

# Selezioniamo i nomi delle colonne numeriche
features_names= [
    'pm2.5', 'pm10',
    'co', 'co_8hr',
    'no2', 'nox', 'no',
    'so2',
    'o3',
    'windspeed', 'winddirec',
    'longitude', 'latitude'
]

df.dropna(axis=0, how='any', subset=features_names+['status'],inplace=True) # EElimina righe con almeno un valore mancante

# Mostra i valori distinti presenti nella colonna 'status'
print(f"\n\nLa nuova dimensione del dataset è:{df.shape}\n\n")
print(f"Valori per ogni classe status:\n{df['status'].value_counts()}\n\n")


X = df[features_names] 
Y = df['status'] # target

# Facciamo lo scaling dei valori presenti nelle colonne numeriche del dataset
# fit_transform standardizza i dati: calcola media e deviazione standard sui dati e li trasforma
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)


# Numero di cluster = numero di categorie status
num_clusters= Y.nunique()
print (f"Status ha {num_clusters} categorie che sono {Y.unique()}")

# KMeans() crea un modello di clustering non supervisionato.
# n_clusters indica quanti gruppi il modello deve cercare nei dati.
kmeans= KMeans(n_clusters=num_clusters, random_state=42) 
# fit_predict addestra il modello sui dati numerici (X) e assegna ogni riga a un cluster
labels= kmeans.fit_predict(X_scaled)

# tabella che mostra quante righe di ogni classe status sono in ciascun cluster
# Utile per capire come i cluster corrispondono alle categorie reali
crosstab= pd.crosstab(Y, labels, colnames=['cluster'])
print(f"Crosstab: {crosstab}") # righe → classi vere (Y), colonne → cluster KMeans, valori → numero di osservazioni


# Per ogni cluster assegna come classe predetta il status più frequente al suo interno (majority vote)
cluster_to_status= crosstab.idxmax(axis=0).to_dict()
print("Mapping cluster --> status: ", cluster_to_status) 

# Trasforma gli ID dei cluster in etichette status
pred_status= pd.Series(labels, index=df.index).map(cluster_to_status)
is_correct= pred_status.eq(Y) # confronta ogni predizione con il valore reale → booleani True/False

# Crea dataframe con numero di predizioni corrette e totali per classe (risultati sulle performance)
summary= (
    # df['status'] è la classe reale di ogni riga del dataset (es: Good, Moderate, ecc.) viene copiata nello stesso ordine del DataFrame originale e serve come verità di riferimento
    # is_correct è una Serie di booleani (True / False) Ogni valore indica se: il cluster assegnato corrisponde allo status reale (True) oppure non corrisponde (False)
    pd.DataFrame({'status': df['status'], 'correct': is_correct})
    # raggruppa per status conta: quante righe sono corrette (True → 1) e quante righe totali ci sono
    .groupby('status')['correct']
    .agg(correct='sum', tot='count')
)


# Aggiungiamo al dataframe "summary" una colonna 'incorrect' 
summary['incorrect']= summary['tot'] - summary['correct']
# Aggiungiamo al dataframe "summary" una colonna 'accuracy_%' dove mettiamo per ogni riga la % di elementi che sono nel cluster corretto
summary['accuracy_%']= (summary['correct'] / summary['tot'] * 100).round(2)

print("Performance per ogni classe di 'status':")
print(summary)

# Calcoliamo e mostriamo la media globale di precisione (media dei valori di "is_correct")
overall_acc= is_correct.mean()
print(f"Accuracy globale (cluster --> status): {overall_acc*100:.2f}%")

# Riduce tutte le feature numeriche a 2 componenti principali (per visualizzazione in 2D)
pca= PCA(n_components=2)
compressed_features= pca.fit_transform(X_scaled)


base_cmap= plt.colormaps["tab10"] # Selezioniamo la base di colori "tab10"
colors= base_cmap.colors[:num_clusters] # Selezioniamo solo i primi "num_clusters" colori di "tab10"
cmap= ListedColormap(colors) # Creiamo una mappa dove assegnamo ogni colore ad un numero

# Usiamo Matplotlib per disegnare il grafico con i cluster
plt.figure(figsize=(8,6))
sc1= plt.scatter( # Disegna un grafico a dispersione. Ogni punto rappresenta una riga del dataset
    compressed_features[:,0], compressed_features[:,1],
    c=labels, # labels contiene l’ID del cluster assegnato da KMeans a ogni punto. Ogni punto viene colorato in base al cluster di appartenenza
    cmap=cmap, # Associa un colore diverso a ciascun cluster
    alpha=0.6, # Formattazione disegno: rende i punti semi-trasparenti
    s=12 # Formattazione disegno: indica la dimensione dei punti
)


# labels è l’array restituito da KMeans che contiene, per ogni riga del dataset, il numero del cluster assegnato (0, 1, 2, …).
# np.unique(labels) estrae i valori unici, cioè l’elenco dei cluster trovati.
clusters_id= np.unique(labels)
colors= [cmap(i) for i in range(num_clusters)] # Assegnamo ad ogni cluster un colore diverso
patches= [mpatches.Patch(color=colors[i], label=f'Cluster {cl}') for i, cl in enumerate(clusters_id)]
# i → indice (0, 1, 2, …), cl → numero reale del cluster (0, 1, 2, …)
# mpatches.Patch(...) crea un quadratino colorato da usare nella legenda
# label=f'Cluster {cl}' testo mostrato nella legenda (es. “Cluster 0”)

# handles=patches → usa i patch creati sopra, title="Labels" → titolo della legenda
# loc='lower left' → posizione in basso a sinistra, frameon=True → disegna il riquadro della legenda
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

plt.title(f'KMeans — PCA 2D (N cluster={num_clusters})')
plt.tight_layout() # Formattazione disegno
plt.show()