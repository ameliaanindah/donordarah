import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("data_bersih.csv")

X = df.drop(["status", "jenis_kelamin"], axis=1)
y = df["status"]

# Split
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model berhasil disimpan sebagai model.pkl")
