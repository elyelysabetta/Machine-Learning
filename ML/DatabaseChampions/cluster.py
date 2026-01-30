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
file = os.path.join(BASE_DIR, 'champions_league.csv') 

if not os.path.exists(file): 
    raise FileNotFoundError(f"File non trovato: {file}")


df = pd.read_csv(file, low_memory=False, na_values=['-', 'NA', 'ND', 'n/a', ''])

# info dataset
print(f"Le dimensioni del dataset sono:{df.shape}\n\n")
print(f"Le prime 5 righe sono:{df.head()}\n\n")
print("Quanti valori validi ci sono per ogni colonna:")
df.info()

colonne_nulle = df.columns[df.isna().all()]
print(f"Colonne completamente nulle: {colonne_nulle.to_list()}") 

# PULIZIA DATAFRAME
df.dropna(axis=1,how='all',inplace=True) 
features_names=['HomeGoals','AwayGoals','HomeShotsOnTarget','AwayShotsOnTarget','HomeShotAccuracy','AwayShotAccuracy']

df.dropna(axis=0,how='any',subset=features_names+['MatchResult'],inplace=True) 
print (f"Valori per ogni classe target: {df['MatchResult'].value_counts()}") 

X = df[features_names] 
Y = df['MatchResult'] 

# standardizzo dati
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)


# Numero di cluster = numero di categorie MatchResult
num_clusters= Y.nunique()
print (f"MatchResult ha {num_clusters} categorie che sono {Y.unique()}")

kmeans= KMeans(n_clusters=num_clusters, random_state=42) 
labels= kmeans.fit_predict(X_scaled) 

crosstab= pd.crosstab(Y, labels, colnames=['cluster'])
print(f"Crosstab: {crosstab}") 

# Per ogni cluster assegna come classe predetta il MatchResult più frequente al suo interno (majority vote)
cluster_to_MatchResult= crosstab.idxmax(axis=0).to_dict()
print("Mapping cluster --> MatchResult ", cluster_to_MatchResult) 


pred_MatchResult= pd.Series(labels, index=df.index).map(cluster_to_MatchResult)
is_correct= pred_MatchResult.eq(Y) 


summary= (
    pd.DataFrame({'MatchResult': df['MatchResult'], 'correct': is_correct})
    .groupby('MatchResult')['correct']
    .agg(correct='sum', tot='count')
)


summary['incorrect']= summary['tot'] - summary['correct'] 
summary['accuracy_%']= (summary['correct'] / summary['tot'] * 100).round(2) 
print(f"Performance per ogni classe di 'MatchResult': {summary}")

overall_acc= is_correct.mean() 
print(f"Accuracy globale (cluster --> MatchResult): {overall_acc*100:.2f}%")

pca= PCA(n_components=2)
compressed_features= pca.fit_transform(X_scaled)


base_cmap= plt.colormaps["tab10"] 
colors= base_cmap.colors[:num_clusters] 
cmap= ListedColormap(colors) 

plt.figure(figsize=(8,6))
sc1= plt.scatter( 
    compressed_features[:,0], compressed_features[:,1],
    c=labels, cmap=cmap, 
    alpha=0.6, s=12 
)

clusters_id= np.unique(labels)
colors= [cmap(i) for i in range(num_clusters)] 
patches= [mpatches.Patch(color=colors[i], label=f'Cluster {cl}') for i, cl in enumerate(clusters_id)]

plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)
plt.title(f'KMeans — PCA 2D (N cluster={num_clusters})')
plt.tight_layout() 
plt.show()