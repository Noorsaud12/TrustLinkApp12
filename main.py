import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# تحميل البيانات
real_df = pd.read_csv("real_users.csv")
fake_df = pd.read_csv("fake_users.csv")

# إضافة label
real_df["label"] = 1
fake_df["label"] = 0

# دمج البيانات
df = pd.concat([real_df, fake_df], ignore_index=True)

# حذف id
df = df.drop(columns=["id"], errors="ignore")

# تجهيز البيانات
X = df.drop("label", axis=1)
y = df["label"]

# تحويل النصوص لأرقام
for col in X.columns:
    X[col] = X[col].astype(str).factorize()[0]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تدريب الموديل
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# حفظ الموديل
pickle.dump(model, open("model.pkl", "wb"))

print("✅ model.pkl created successfully")