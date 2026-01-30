import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

DIR=os.path.dirname(os.path.abspath(__file__))
file=os.path.join(DIR,'Social.csv')

if not os.path.exists(file):
    raise FileNotFoundError(f"{file} non trovato")

# dataframe
df=pd.read_csv(file,low_memory=False,na_values=['NA','ND','n/a','-',''])

# info dataframe
print(f"Dimensione dataframe: {df.shape}")
print(f"Prime 5 righe: {df.head()}\n")
print ("Info colonne:")
df.info()

colonne_nulle=df.columns[df.isna().all()]
print(f"\nColonne nulle: {colonne_nulle.tolist()}")

df.dropna(axis=1,how='all',inplace=True)

encoder=LabelEncoder() # trasformo object in int
df['Gender']=encoder.fit_transform(df['Gender'])
df['Academic_Level']=encoder.fit_transform(df['Academic_Level'])
df['Country']=encoder.fit_transform(df['Country'])
df['Affects_Academic_Performance']=encoder.fit_transform(df['Affects_Academic_Performance'])
df['Relationship_Status']=encoder.fit_transform(df['Relationship_Status'])
df['Most_Used_Platform']=encoder.fit_transform(df['Most_Used_Platform'])

features=df.select_dtypes(include=[np.number]).columns.tolist()
features.remove('Student_ID')
features.remove('Age')

df.dropna(axis=0,how='any',subset=features+['Age'],inplace=True)

X=df[features]
Y=df['Age']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=31)

print(f"\nDimensioni X_train: {X_train.shape}")
print(f"Dimensioni X_test: {X_test.shape}")

model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

r=Y_test.max()-Y_test.min() # range
mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
err=(rmse/r)*100

print(f"Range: {r} \nMSE: {mse:.2f} \nRMSE: {rmse:.2f} \n% errore: {err:.2f}")

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
    label="Valori ideali"
)
plt.title("Regressione Lineare")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
