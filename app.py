import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
 
sb.set_theme(style="whitegrid", palette="Set2")

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(rgba(17, 32, 45, 0.58), rgba(17, 32, 45, 0.64)),
            url("https://images.unsplash.com/photo-1522202176988-66273c2fd55f?auto=format&fit=crop&w=1800&q=85");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #ffffff;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    header[data-testid="stHeader"] {
        background: rgba(17, 32, 45, 0.88);
        border-bottom: 1px solid rgba(156, 181, 203, 0.58);
        box-shadow: 0 4px 18px rgba(28, 42, 68, 0.08);
        backdrop-filter: blur(10px);
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        opacity: 1;
        visibility: visible;
    }
    [data-testid="stToolbar"] button,
    header[data-testid="stHeader"] button {
        color: #ffffff;
    }
    h1 {
        color: #ffffff;
        letter-spacing: 0;
        padding-bottom: 0.35rem;
    }
    h2, h3 {
        color: #ffffff;
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] {
        background: rgba(17, 32, 45, 0.92);
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(156, 181, 203, 0.56);
        border-radius: 8px;
        box-shadow: 0 10px 24px rgba(28, 42, 68, 0.08);
        background: rgba(17, 32, 45, 0.72);
        backdrop-filter: blur(8px);
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #b7c8da;
        box-shadow: 0 12px 30px rgba(28, 42, 68, 0.12);
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #ffffff;
        line-height: 1.65;
    }
    [data-testid="stMarkdownContainer"],
    label,
    .stSelectbox label,
    .stSlider label {
        color: #ffffff;
    }
    [data-testid="stDataFrame"],
    [data-testid="stChart"] {
        border: 1px solid #dce3ee;
        border-radius: 8px;
        overflow: hidden;
        background: rgba(17, 32, 45, 0.68);
    }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stAlert"]) {
        margin-top: 0.75rem;
    }
    .stButton button,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 8px;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stSlider [data-baseweb="slider"] {
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #457b9d;
        box-shadow: 0 0 0 3px rgba(69, 123, 157, 0.12);
    }
    [data-testid="stAlert"] {
        border-radius: 8px;
    }
    hr {
        margin: 2rem 0 1.25rem 0;
        border-color: #dce3ee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Student Performance Dashboard")

st.markdown("""
### What is this?
This app helps understand **student behavior and performance** using simple visuals and predictions.

You do not need to know machine learning. Just explore the charts below.

- More activity, like raising hands and using resources, usually means better performance
- Fewer absences usually means better outcomes
""")

data = pd.read_csv("AI-Data.csv")

with st.container(border=True):
    st.subheader("Dataset Preview")
    st.dataframe(data.head(), width="stretch")

data_processed = data.copy()

for col in data_processed.columns:
    if data_processed[col].dtype in ("object", "category", "string"):
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])

st.divider()
with st.container(border=True):
    st.subheader("Correlation Heatmap")

    st.markdown("""
**How to read this:**
- Values closer to **1** = strong relationship
- Values closer to **0** = weak relationship
- Example: If two things have high value, they often happen together
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
    st.pyplot(fig, width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Feature Relationships")

    st.markdown("""
This shows how different student activities relate to performance.

- Each color = different performance group
- Clusters = similar student behavior
""")

    pairplot_fig = sb.pairplot(data, hue="Class")
    st.pyplot(pairplot_fig.fig, width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Explore Data")

    option = st.selectbox("Choose what you want to see", [
        "Distribution of Study Activity",
        "Performance vs Activity",
        "Class Distribution",
    ])

    if option == "Distribution of Study Activity":
        st.markdown("Shows how active students are overall")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sb.histplot(data_processed['raisedhands'], kde=True, ax=ax)
        ax.set_title("How often students participate (raised hands)")
        ax.set_xlabel("Raised hands")
        ax.set_ylabel("Number of students")
        st.pyplot(fig, width="stretch")

    elif option == "Performance vs Activity":
        st.markdown("Compare activity levels across performance groups")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sb.boxplot(x=data_processed['Class'], y=data_processed['raisedhands'], ax=ax)
        ax.set_title("Participation vs Performance")
        ax.set_xlabel("Class")
        ax.set_ylabel("Raised hands")
        st.pyplot(fig, width="stretch")

    elif option == "Class Distribution":
        st.markdown("Shows how many students fall into each category")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sb.countplot(x=data_processed['Class'], ax=ax)
        ax.set_title("Performance Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of students")
        st.pyplot(fig, width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Model Performance")

    st.markdown("""
We trained different models to predict student performance.

- Higher accuracy = better predictions
- This helps identify which model works best
""")

model_features = [
    "raisedhands",
    "VisITedResources",
    "Discussion",
    "StudentAbsenceDays",
]

X = pd.get_dummies(data[model_features], columns=["StudentAbsenceDays"], dtype=float)
y = data["Class"]

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

with st.container(border=True):
    acc_col1, acc_col2, acc_col3 = st.columns(3)
    acc_col1.write(f"Decision Tree Accuracy: {round(acc_dt, 3)}")
    acc_col2.write(f"Random Forest Accuracy: {round(acc_rf, 3)}")
    acc_col3.write(f"Logistic Regression Accuracy: {round(acc_lr, 3)}")

st.divider()
with st.container(border=True):
    st.subheader("Model Comparison")
    acc_df = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest", "Logistic"],
        "Accuracy": [acc_dt, acc_rf, acc_lr]
    })
    st.bar_chart(acc_df.set_index("Model"), width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Where Predictions Go Wrong")

    st.markdown("""
This shows how often the model makes correct vs incorrect predictions.
""")

    cm = confusion_matrix(y_test, pred_rf)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sb.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title("Confusion Matrix (Random Forest)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    st.pyplot(fig, width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("What matters most?")

    st.markdown("""
These are the most important factors affecting performance.
""")

    importance = model_rf.feature_importances_
    features = X.columns

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sb.barplot(x="Importance", y="Feature", data=imp_df, ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig, width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Try Your Own Prediction")

    st.markdown("""
Enter student activity details to predict performance.

- More activity means higher chance of better performance
- Less absence usually means better results
""")

    input_col, result_col = st.columns([1.15, 0.85])

    with input_col:
        raised = st.slider("Raised Hands (Participation)", 0, 100)
        resources = st.slider("Visited Resources (Study Material Usage)", 0, 100)
        discussion = st.slider("Discussion Participation", 0, 100)
        absence = st.selectbox("Absence", ["Under-7", "Above-7"])

    input_data = pd.DataFrame([{
        "raisedhands": raised,
        "VisITedResources": resources,
        "Discussion": discussion,
        f"StudentAbsenceDays_{absence}": 1.0,
    }])

    input_data = input_data.reindex(columns=X.columns, fill_value=0.0)
    input_scaled = scaler.transform(input_data)

    prediction = model_rf.predict(input_scaled)

    with result_col:
        st.success(f"Predicted Performance: {prediction[0]}")

        st.markdown("""
### Simple Insight:
- Stay active in class
- Use learning resources
- Avoid too many absences

These greatly improve performance.
""")
