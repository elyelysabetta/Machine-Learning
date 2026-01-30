# predico HomeGoals
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

DIR=os.path.dirname(os.path.abspath(__file__))
file=os.path.join(DIR,'champions_league.csv')

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

encoder=LabelEncoder()
df['MatchResult']=encoder.fit_transform(df['MatchResult'])

# PULIZIA
df.dropna(axis=1,how='all',inplace=True)
features=['HomeShots','AwayShots','HomeShotsOnTarget','AwayShotsOnTarget','HomeShotAccuracy','AwayShotAccuracy','AwayGoals','MatchResult']
df.dropna(axis=0,how='any',subset=features+['HomeGoals'],inplace=True)

X=df[features]
Y=df['HomeGoals']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=33)
print (f"\nDimensioni X_train: {X_train.shape}")
print (f"Dimensioni X_test: {X_test.shape}")

model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

r=Y_test.max()-Y_test.min() # range
mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
err=(rmse/r)*100

print(f"Range: {r:.2f} \nMSE: {mse:.2f} \nRMSE: {rmse:.2f} \n% errori: {err:.2f}")

plt.scatter(
    Y_test,Y_pred,
    alpha=0.3,
    color='green',
    label="Valori predetti"
)

plt.plot(
    [Y_test.min(),Y_test.max()],
    [Y_test.min(),Y_test.max()],
    color='blue',
    label="Predizione perfetta"
)
plt.title("Regressione lineare")
plt.legend()
plt.grid(alpha=0.2)
plt.show()