# predico tatal_price
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
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

# PULISCO DATAFRAME
df.dropna(axis=1,how='all',inplace=True)

encoder=LabelEncoder()
df['travel_type']=encoder.fit_transform(df['travel_type'])
df['season']=encoder.fit_transform(df['season'])

features=df.select_dtypes(include=[np.number]).columns.tolist()
features.remove('booking_id')
features.remove('total_price')

df.dropna(axis=0,how='any',subset=features+['total_price'],inplace=True)
print (f"\nNuove dimensioni: {df.shape}")

X=df[features]
Y=df['total_price']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=33)

print(f"\nDimensioni X_train: {X_train.shape}")
print(f"Dimensioni X_test: {X_test.shape}")

model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

r=Y_test.max()-Y_train.min() # range
mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
err=(rmse/r)*100

print(f"\nRange: {r:.2f} \nMSE: {mse:.2f} \nRMSE: {rmse:.2f} \n% errori: {err:.2f}")

plt.scatter(
    Y_test,Y_pred,
    alpha=0.3,
    color='red',
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




