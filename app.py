import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# تحميل الموديل
model = pickle.load(open("model.pkl", "rb"))

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

# واجهة المشروع
st.title("🔍 Fake vs Real User Detection")

# إدخال اسم المستخدم
username = st.text_input("Enter Username")

if st.button("Search"):

    user_data = df[df["name"].astype(str).str.lower() == username.strip().lower()]

    if user_data.empty:
        st.error("❌ User not found")

    else:
        st.success("✅ User found")

        # تجهيز البيانات
        X = user_data.drop("label", axis=1)

        for col in X.columns:
            X[col] = X[col].astype(str).factorize()[0]

        # التوقع
        prediction = model.predict(X)

        if prediction[0] == 1:
            st.success("✅ REAL USER")
        else:
            st.error("❌ FAKE USER")

        # شارت
        st.subheader("📊 User Statistics")

        numeric_df = user_data.select_dtypes(include=['int64', 'float64'])

        if numeric_df.shape[1] > 0:

            data = numeric_df.iloc[0]

            fig, ax = plt.subplots()

            ax.bar(data.index[:5], data.values[:5])

            plt.xticks(rotation=45)

            st.pyplot(fig)