import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

DIR=os.path.dirname(os.path.abspath(__file__))
file=os.path.join(DIR,'travel.csv')

if not os.path.exists(file):
    raise FileNotFoundError(f"{file} non trovato")

# leggo file e salvo i dati in dataframe
df=pd.read_csv(file,low_memory=False,na_values=['NA','ND','n/a','-',''])

# visualizzo info dataframe
print (f"Dimensione dataframe: {df.shape}")
print (f"Prime 5 righe: {df.head()}\n")
print ("Informazioni colonne:")
df.info()

colonne_nulle=df.columns[df.isna().all()]
print(f"\nColonne vuote: {colonne_nulle.tolist()}")

# PULIZIA
df.dropna(axis=1,how='all',inplace=True)
df.dropna(axis=0,how='any',subset='cancelled',inplace=True)

print (f"\nValori per ogni classe del target: {df['cancelled'].value_counts()}") # molto sbilanciata

encoder=LabelEncoder()
df['travel_type']=encoder.fit_transform(df['travel_type'])
df['transport']=encoder.fit_transform(df['transport'])
df['season']=encoder.fit_transform(df['season'])
df['travel_insurance']=encoder.fit_transform(df['travel_insurance'])

features=df.select_dtypes(include=[np.number]).columns.tolist()
features.remove('booking_id')
features.remove('age')

X=df[features]
Y=df['cancelled']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=33)

print(f"\nDimensioni X_train: {X_train.shape}")
print(f"Dimensioni X_test: {X_test.shape}")

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=DecisionTreeClassifier()
model.fit(X_train_scaled,Y_train) # addestro il modello

Y_pred=model.predict(X_test_scaled)
print(f"\nAccuratezza modello: {accuracy_score(Y_test,Y_pred)}")

plt.figure(figsize=(20,10))
plot_tree(
    model,
    max_depth=3,
    feature_names=X.columns,
    class_names=[str(classe) for classe in sorted(Y.unique())],
    filled=True,
    fontsize=5
)
plt.title("Albero decisionale")
plt.show()

cm=confusion_matrix(Y_test,Y_pred,normalize="true")
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sorted(Y.unique()))
disp.plot(cmap='Blues')
plt.title("Matrice di confusione")
plt.legend()
plt.show()