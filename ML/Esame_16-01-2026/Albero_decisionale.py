import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

DIR=os.path.dirname(os.path.abspath(__file__))
file=os.path.join(DIR,'drug200.csv')

if not os.path.exists(file):
    raise FileNotFoundError(f"{file} non trovato")

# leggo il file .csv e carico i dati in un dataframe
df=pd.read_csv(
    file,
    low_memory=False,
    na_values=['NA','ND','n/a','-',''],
    nrows=100000
)

print(f"Dimensione dataset: {df.shape}")
print(f"Le prime 5 righe: {df.head()}")
print("\nQuanti valori ci sono per ogni colonna: ")
df.info()

colonne_nulle=df.columns[df.isna().all()]
print(f"\nColonne completanente nulle: {colonne_nulle.to_list()}") # non ho colonne vuote

df.dropna(axis=1,how='all',inplace=True) # elimino le colonne completamente vuote
df.dropna(axis=0,how='any',subset='Drug',inplace=True) # elimino le righe con target 'Drug' nullo

print (f"\nNuove dimensioni del dataset: {df.shape}") # non sono cambiate le dimensioni
print(f"Valori per ogni classe Drug {df['Drug'].value_counts()}\n") # sbilanciato

# trasformo le etichette stringhe (sex,BP,cholesterol) in numeri 
lencoder=LabelEncoder()
df['Sex']=lencoder.fit_transform(df['Sex'])
df['BP']=lencoder.fit_transform(df['BP'])
df['Cholesterol']=lencoder.fit_transform(df['Cholesterol'])

print("Nuovi valori: ") # ho cambiato formato
df.info()


features_names=df.select_dtypes(include=[np.number]).columns.tolist()

X=df[features_names] #features 
Y=df['Drug'] # target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3) # 30% test e 70% training

print(f"Numero righe e colonne di X_train: {X_train.shape}")
print(f"Numero righe e colonne di X_test: {X_test.shape}")

# standardizzo i dati
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=DecisionTreeClassifier() 
model.fit(X_train_scaled,Y_train) # adestramento modello

Y_pred=model.predict(X_test_scaled)
print (f"Acuratezza: {accuracy_score(Y_test,Y_pred)}") # perfetto
print(f"Features utilizzate: {X.columns.to_list()}")

plt.figure(figsize=(20,10))
plot_tree(
    model,
    max_depth=3,
    class_names=[str(classi) for classi in sorted(Y.unique())],
    feature_names=X.columns,
    filled=True, # colore
    fontsize=5
)
plt.title("Albero decisionale")
plt.show()

cm=confusion_matrix(Y_test,Y_pred,normalize="true")
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sorted(Y.unique()))
disp.plot(cmap='Purples')
plt.title("Matrice di confuzione")
plt.show()







