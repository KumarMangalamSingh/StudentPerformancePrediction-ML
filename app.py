import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Dashboard")

st.markdown("""
### 👋 What is this?
This app helps understand **student behavior and performance** using simple visuals and predictions.

👉 You don’t need to know machine learning — just explore the charts below.

- More activity (like raising hands, using resources) → usually better performance  
- Fewer absences → better outcomes  
""")

data = pd.read_csv("AI-Data.csv")

st.subheader("📄 Dataset Preview")
st.dataframe(data.head())

data_processed = data.copy()

for col in data_processed.columns:
    if data_processed[col].dtype == 'object':
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])

st.subheader("🔥 Correlation Heatmap")

st.markdown("""
👉 **How to read this:**
- Values closer to **1** = strong relationship  
- Values closer to **0** = weak relationship  
- Example: If two things have high value → they often happen together  
""")

corr = data_processed.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(8, 6))
sb.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 8},
    ax=ax
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
st.pyplot(fig)

st.subheader("🔗 Feature Relationships")

st.markdown("""
👉 This shows how different student activities relate to performance.

- Each color = different performance group  
- Clusters = similar student behavior  
""")

pairplot_fig = sb.pairplot(data, hue="Class")
st.pyplot(pairplot_fig.fig)

st.subheader("📊 Explore Data")

option = st.selectbox("Choose what you want to see", [
    "Distribution of Study Activity",
    "Performance vs Activity",
    "Class Distribution",
])

if option == "Distribution of Study Activity":
    st.markdown("👉 Shows how active students are overall")
    fig, ax = plt.subplots()
    sb.histplot(data_processed['raisedhands'], kde=True, ax=ax)
    ax.set_title("How often students participate (raised hands)")
    st.pyplot(fig)

elif option == "Performance vs Activity":
    st.markdown("👉 Compare activity levels across performance groups")
    fig, ax = plt.subplots()
    sb.boxplot(x=data_processed['Class'], y=data_processed['raisedhands'], ax=ax)
    ax.set_title("Participation vs Performance")
    st.pyplot(fig)

elif option == "Class Distribution":
    st.markdown("👉 Shows how many students fall into each category")
    fig, ax = plt.subplots()
    sb.countplot(x=data_processed['Class'], ax=ax)
    ax.set_title("Performance Distribution")
    st.pyplot(fig)

st.subheader("🤖 Model Performance")

st.markdown("""
👉 We trained different models to predict student performance.

- Higher accuracy = better predictions  
- This helps identify which model works best  
""")

X = data_processed.drop("Class", axis=1)
y = data_processed["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_dt = DecisionTreeClassifier()
model_rf = RandomForestClassifier()
model_lr = LogisticRegression(max_iter=1000)

model_dt.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)

pred_dt = model_dt.predict(X_test)
pred_rf = model_rf.predict(X_test)
pred_lr = model_lr.predict(X_test)

acc_dt = accuracy_score(y_test, pred_dt)
acc_rf = accuracy_score(y_test, pred_rf)
acc_lr = accuracy_score(y_test, pred_lr)

st.write("Decision Tree Accuracy:", round(acc_dt, 3))
st.write("Random Forest Accuracy:", round(acc_rf, 3))
st.write("Logistic Regression Accuracy:", round(acc_lr, 3))

st.subheader("📈 Model Comparison")
acc_df = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Logistic"],
    "Accuracy": [acc_dt, acc_rf, acc_lr]
})
st.bar_chart(acc_df.set_index("Model"))

st.subheader("📉 Where Predictions Go Wrong")

st.markdown("""
👉 This shows how often the model makes correct vs incorrect predictions.
""")

cm = confusion_matrix(y_test, pred_rf)

fig, ax = plt.subplots()
sb.heatmap(cm, annot=True, fmt="d", ax=ax)
ax.set_title("Confusion Matrix (Random Forest)")
st.pyplot(fig)

st.subheader("⭐ What matters most?")

st.markdown("""
👉 These are the most important factors affecting performance.
""")

importance = model_rf.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sb.barplot(x="Importance", y="Feature", data=imp_df, ax=ax)
st.pyplot(fig)

st.subheader("🔮 Try Your Own Prediction")

st.markdown("""
👉 Enter student activity details to predict performance.

- More activity → higher chance of better performance  
- Less absence → better results  
""")

raised = st.slider("Raised Hands (Participation)", 0, 100)
resources = st.slider("Visited Resources (Study Material Usage)", 0, 100)
discussion = st.slider("Discussion Participation", 0, 100)
absence = st.selectbox("Absence", ["Under-7 days", "Above-7 days"])

absence_val = 1 if absence == "Under-7 days" else 0

input_data = np.array([[raised, resources, discussion, absence_val]])

if input_data.shape[1] != X.shape[1]:
    input_data = np.pad(input_data, ((0, 0), (0, X.shape[1] - input_data.shape[1])))

input_scaled = scaler.transform(input_data)

prediction = model_rf.predict(input_scaled)

st.success(f"🎯 Predicted Performance: {prediction[0]}")

st.markdown("""
### 🧠 Simple Insight:
- Stay active in class  
- Use learning resources  
- Avoid too many absences  

👉 These greatly improve performance!
""")