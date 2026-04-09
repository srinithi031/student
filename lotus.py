import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# ---------------- Sidebar Navigation ----------------
page = st.sidebar.selectbox(
    "📌 Select Page",
    ["🏠 Home", "📊 Analysis", "🤖 Model Training", "🔮 Predict Future Marks"]
)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("🎓 Student Performance Predictor")
    st.markdown("""
    This app predicts **Semester VI marks** based on previous semesters.

    ### 🔹 Features:
    - 📊 Data Analysis
    - 🤖 Model Training
    - 🔮 Future Prediction
    """)

# ---------------- ANALYSIS ----------------
elif page == "📊 Analysis":
    st.header("📊 Data Analysis")

    # Sample dataset (you can replace with CSV upload)
    data = {
        'Sem I': np.random.randint(50, 100, 50),
        'Sem II': np.random.randint(50, 100, 50),
        'Sem III': np.random.randint(50, 100, 50),
        'Sem IV': np.random.randint(50, 100, 50),
        'Sem V': np.random.randint(50, 100, 50),
        'Sem VI': np.random.randint(50, 100, 50),
    }

    df = pd.DataFrame(data)
    st.write(df.head())

    st.subheader("📈 Correlation Heatmap")
    st.write(df.corr())

# ---------------- MODEL TRAINING ----------------
elif page == "🤖 Model Training":
    st.header("🤖 Train Model")

    # Sample dataset
    data = {
        'Sem I': np.random.randint(50, 100, 100),
        'Sem II': np.random.randint(50, 100, 100),
        'Sem III': np.random.randint(50, 100, 100),
        'Sem IV': np.random.randint(50, 100, 100),
        'Sem V': np.random.randint(50, 100, 100),
        'Internship_Code': np.random.randint(0, 2, 100),
        'Sex_Code': np.random.randint(0, 2, 100),
        'Sem VI': np.random.randint(50, 100, 100),
    }

    df = pd.DataFrame(data)

    X = df.drop("Sem VI", axis=1)
    y = df["Sem VI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.success(f"✅ Model Trained! RMSE: {round(rmse,2)}")

    # Save model
    st.session_state['best_model'] = model

# ---------------- PREDICTION ----------------
elif page == "🔮 Predict Future Marks":

    st.header("🎯 Predict Future Semester (Sem VI) Marks")
    st.markdown("Enter marks for Sem I to V (out of 100)")

    col1, col2, col3 = st.columns(3)

    with col1:
        sem1 = st.number_input("Sem I", 0.0, 100.0, 75.0)
        sem2 = st.number_input("Sem II", 0.0, 100.0, 75.0)
        sem3 = st.number_input("Sem III", 0.0, 100.0, 75.0)

    with col2:
        sem4 = st.number_input("Sem IV", 0.0, 100.0, 75.0)
        sem5 = st.number_input("Sem V", 0.0, 100.0, 75.0)
        gender = st.selectbox("Gender", ['female', 'male'])

    with col3:
        internship = st.selectbox("Internship", ['Yes', 'No'])
        predict_btn = st.button("🚀 Predict", use_container_width=True)

    gender_code = 0 if gender == 'female' else 1
    internship_code = 1 if internship == 'Yes' else 0

    input_data = pd.DataFrame([[sem1, sem2, sem3, sem4, sem5, internship_code, gender_code]],
                              columns=['Sem I', 'Sem II', 'Sem III', 'Sem IV', 'Sem V', 'Internship_Code', 'Sex_Code'])

    if 'best_model' not in st.session_state:
        st.warning("⚠️ Train the model first from Model Training page")
        st.stop()

    if predict_btn:
        model = st.session_state['best_model']
        prediction = model.predict(input_data)[0]
        prediction = round(max(0, min(100, prediction)), 2)

        st.success(f"📌 Predicted Sem VI Marks: {prediction} / 100")

        # Performance message
        if prediction >= 75:
            st.balloons()
            st.info("🌟 Excellent Performance Expected")
        elif prediction >= 60:
            st.info("👍 Good Performance")
        else:
            st.warning("⚠️ Needs Improvement")

        # -------- Line Chart --------
        semesters = ['Sem I', 'Sem II', 'Sem III', 'Sem IV', 'Sem V', 'Sem VI']
        marks = [sem1, sem2, sem3, sem4, sem5, prediction]

        fig, ax = plt.subplots()
        ax.plot(semesters, marks, marker='o')
        ax.set_ylim(0, 100)
        ax.set_title("Performance Trend")
        st.pyplot(fig)

        # -------- Bar Chart --------
        fig2, ax2 = plt.subplots()
        ax2.bar(semesters, marks)
        ax2.set_ylim(0, 100)
        ax2.set_title("Marks Comparison")
        st.pyplot(fig2)

        st.subheader("📋 Summary")
        st.write(f"Gender: {gender} | Internship: {internship}")
        st.write(f"Marks: {sem1}, {sem2}, {sem3}, {sem4}, {sem5}")

