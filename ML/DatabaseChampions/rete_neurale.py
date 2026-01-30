import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

DIR=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(DIR,'champions_league.csv')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} non trovalto")

df = pd.read_csv(file_path, na_values=["NA", "-", "na", "N/A", "ND", ""], nrows=1000000)

# visualizzo informazioni file 
print(f"Dimensioni file: {df.shape}")
print (f"Prime 5 righe: {df.head()}")
print ("\nInfo colonne: ")
df.info()

# visualizzo colonne nulle 
colonne_nulle=df.columns[df.isna().all()]
print(f"\nColonne nulle: {colonne_nulle.tolist()}") # non ci sono colonne nulle

target = "MatchResult"
df = df.dropna(subset=[target])

print(f"\nVisualizzo i valori dello 'MatchResult': {df[target].unique()}")

class_names = df[target].unique()
num_classes = len(class_names)



df = df.dropna(axis=1, how="all")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

df = df.dropna(subset=num_cols, how="any")

feature_cols =['HomeGoals','AwayGoals','HomeShotsOnTarget','AwayShotsOnTarget','HomeShotAccuracy','AwayShotAccuracy']

print(df.info())

print("\nConteggio classi nel cleaned DataFrame: ")
print(df[target].value_counts().sort_index())

print("\nDimensioni dopo il cleaning: ", df.shape)

test_frac = 0.2 
val_frac = 0.1 

# train/test
train_df, test_df = train_test_split(df, test_size=test_frac, random_state=42)

# train/val
train_df, val_df = train_test_split(train_df, test_size=val_frac, random_state=42)

# Controllo se tutte le classi sono presenti in ogni df
all_classes = set(np.unique(df[target]))
train_classes = set(np.unique(train_df[target]))
test_classes = set(np.unique(test_df[target]))

missing_train = all_classes - train_classes
missing_test = all_classes - test_classes

print(f"\nMissing class train: {missing_train}")
print(f"\nMissing class test: {missing_test}")

print(f"Classes prsent in the train_df: {sorted(train_df[target].unique())}")
print(f"Classes prsent in the test_df: {sorted(test_df[target].unique())}")


X_train = train_df[num_cols]
X_val = val_df[num_cols]
X_test = test_df[num_cols]

y_train = train_df[target]
y_val = val_df[target]
y_test = test_df[target]

# Trasforma classi categoriche → interi
encoder = LabelEncoder()
encoder.fit(y_train)
y_train_idx = encoder.transform(y_train)
y_val_idx = encoder.transform(y_val)
y_test_idx = encoder.transform(y_test)

y_train_oh = to_categorical(y_train_idx, num_classes=num_classes)
y_val_oh = to_categorical(y_val_idx, num_classes=num_classes)
y_test_oh = to_categorical(y_test_idx, num_classes=num_classes)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"\nTraining set dimensions: {X_train.shape}")
print(f"\nTest set dimensions: {X_test.shape}")

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)), 
    layers.Dense(16, activation="relu"), 
    layers.Dense(8, activation="relu"), 
    layers.Dense(num_classes, activation="softmax") 
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy", 
    metrics=["accuracy"], 
)

history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=50,
    batch_size=128,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

y_pred = model.predict(X_test, verbose=0).argmax(axis=1)


cm = confusion_matrix(y_test_idx, y_pred, normalize="true")
plt.figure(figsize=(10,8))

sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()