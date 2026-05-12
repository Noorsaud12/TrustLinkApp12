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

# حذف id إذا موجود
df = df.drop(columns=["id"], errors="ignore")

# واجهة المشروع
st.title("🔍 Fake vs Real User Detection")

# إدخال اسم المستخدم
username = st.text_input("Enter Username")

if st.button("Search"):

    # البحث عن المستخدم
    user_data = df[
        df["name"].astype(str).str.lower()
        == username.strip().lower()
    ]

    # إذا المستخدم غير موجود
    if user_data.empty:
        st.error("❌ User not found")

    else:
        st.success("✅ User found")

        # الأعمدة المستخدمة بالتدريب
        features = [
            "followers_count",
            "friends_count",
            "statuses_count",
            "favourites_count"
        ]

        # تجهيز البيانات
        X = user_data[features]

        # التوقع
        prediction = model.predict(X)

        # النتيجة
        if prediction[0] == 1:
            st.success("✅ REAL USER")
        else:
            st.error("❌ FAKE USER")

        # التأكد من وجود الأعمدة
        available_features = [
            f for f in features if f in user_data.columns
        ]

        # الرسم البياني
        if len(available_features) > 0:

            values = user_data[available_features].iloc[0]

            fig, ax = plt.subplots()

            ax.bar(available_features, values)

            ax.set_title("User Statistics")

            st.pyplot(fig)

        else:
            st.warning("Required columns not found")
