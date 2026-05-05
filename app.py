import streamlit as st
import sys
import site
from pathlib import Path

USER_SITE_CANDIDATES = []
try:
    USER_SITE_CANDIDATES.append(site.getusersitepackages())
except Exception:
    pass

local_appdata = Path.home() / "AppData" / "Local" / "Packages"
USER_SITE_CANDIDATES.extend(
    local_appdata.glob("PythonSoftwareFoundation.Python.3.13_*\\LocalCache\\local-packages\\Python313\\site-packages")
)

for candidate in USER_SITE_CANDIDATES:
    candidate_str = str(candidate)
    if candidate_str and candidate_str not in sys.path and Path(candidate_str).exists():
        sys.path.append(candidate_str)

import pandas as pd
try:
    import seaborn as sb
except ModuleNotFoundError:
    class SimpleSeabornFallback:
        def set_theme(self, style="whitegrid", palette="Set2"):
            plt.style.use("seaborn-v0_8-whitegrid")

        def barplot(self, data, x, y, ax, hue=None, color=None):
            if hue is None:
                ax.bar(data[x].astype(str), data[y], color=color or "#4c78a8")
                return

            pivot = data.pivot(index=x, columns=hue, values=y).fillna(0)
            categories = list(pivot.index.astype(str))
            hue_levels = list(pivot.columns.astype(str))
            width = 0.8 / max(len(hue_levels), 1)
            positions = list(range(len(categories)))

            for idx, hue_level in enumerate(hue_levels):
                offsets = [pos - 0.4 + width / 2 + idx * width for pos in positions]
                ax.bar(offsets, pivot[hue_level].tolist(), width=width, label=hue_level)

            ax.set_xticks(positions)
            ax.set_xticklabels(categories)
            ax.legend(title=hue)

        def histplot(self, values, kde=True, ax=None):
            ax.hist(values.dropna(), bins=10, color="#4c78a8", alpha=0.85, edgecolor="white")

        def countplot(self, x, ax=None):
            counts = pd.Series(x).value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")

    sb = SimpleSeabornFallback()
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

# st.markdown("""
# ### What is this?
# This app helps understand **student behavior and performance** using simple visuals and predictions.

# You do not need to know machine learning. Just explore the charts below.

# - More activity, like raising hands and using resources, usually means better performance
# - Fewer absences usually means better outcomes
# """)

csv_path = Path(__file__).with_name("AI-Data.csv")
CSV_COLUMNS = [
    "StudentName",
    "gender",
    "StageID",
    "SectionID",
    "Topic",
    "QuestionsAnswered",
    "ActiveParticipation",
    "UnitTestMarks",
    "UnitTestMaxMarks",
    "PeriodicTestMarks",
    "PeriodicTestMaxMarks",
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
    "Class",
]

DEFAULT_CATEGORY_OPTIONS = {
    "gender": ["F", "M"],
    "StageID": ["lowerlevel", "MiddleSchool", "HighSchool"],
    "SectionID": ["A", "B", "C"],
    "Topic": [
        "English",
        "Hindi",
        "Mathematics",
        "Science",
        "Social Science",
        "Computer Science",
        "Physics",
        "Chemistry",
        "Biology",
        "History",
        "Geography",
        "Economics",
        "Political Science",
        "Accountancy",
        "Business Studies",
    ],
    "ParentAnsweringSurvey": ["Yes", "No"],
    "ParentschoolSatisfaction": ["Good", "Bad"],
    "StudentAbsenceDays": ["Under-7", "Above-7"],
    "Class": ["H", "M", "L"],
}

def merged_category_options(current_values, default_values):
    merged = list(default_values)
    for value in current_values:
        if value not in merged:
            merged.append(value)
    return merged


def load_active_data(path):
    loaded = pd.read_csv(path, comment="#")
    loaded = loaded.reindex(columns=CSV_COLUMNS)
    numeric_columns = [
        "QuestionsAnswered",
        "ActiveParticipation",
        "UnitTestMarks",
        "UnitTestMaxMarks",
        "PeriodicTestMarks",
        "PeriodicTestMaxMarks",
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion",
    ]
    for col in numeric_columns:
        loaded[col] = pd.to_numeric(loaded[col], errors="coerce")
    loaded["UnitTestPercent"] = (loaded["UnitTestMarks"] / loaded["UnitTestMaxMarks"].replace(0, pd.NA)) * 100
    loaded["PeriodicTestPercent"] = (loaded["PeriodicTestMarks"] / loaded["PeriodicTestMaxMarks"].replace(0, pd.NA)) * 100
    loaded = loaded[loaded["Class"].isin(DEFAULT_CATEGORY_OPTIONS["Class"])]
    required_numeric = [
        "QuestionsAnswered",
        "ActiveParticipation",
        "UnitTestMarks",
        "UnitTestMaxMarks",
        "PeriodicTestMarks",
        "PeriodicTestMaxMarks",
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion",
    ]
    loaded = loaded.dropna(subset=required_numeric)
    return loaded


data = load_active_data(csv_path)
model_ready = len(data) >= 5 and data["Class"].nunique() >= 2


def benchmark_pair(series, default_low=30.0, default_high=70.0):
    if series.dropna().empty:
        return {"low": default_low, "high": default_high}
    return {
        "low": float(series.quantile(0.25)),
        "high": float(series.quantile(0.75)),
    }

NUMERIC_BENCHMARKS = {
    "QuestionsAnswered": benchmark_pair(data["QuestionsAnswered"], 2.0, 8.0),
    "ActiveParticipation": benchmark_pair(data["ActiveParticipation"]),
    "UnitTestPercent": benchmark_pair(data["UnitTestPercent"], 35.0, 75.0),
    "PeriodicTestPercent": benchmark_pair(data["PeriodicTestPercent"], 35.0, 75.0),
    "raisedhands": benchmark_pair(data["raisedhands"]),
    "VisITedResources": benchmark_pair(data["VisITedResources"]),
    "AnnouncementsView": benchmark_pair(data["AnnouncementsView"]),
    "Discussion": benchmark_pair(data["Discussion"]),
}

DEFAULT_VISITED_RESOURCES = int(round(data["VisITedResources"].mean())) if not data.empty else 50
DEFAULT_ANNOUNCEMENTS_VIEW = int(round(data["AnnouncementsView"].mean())) if not data.empty else 5


def score_to_prediction_label(score):
    if score >= 75:
        return "H"
    if score >= 50:
        return "M"
    return "L"


def factor_status(value, metric_name):
    if value < NUMERIC_BENCHMARKS[metric_name]["low"]:
        return "needs_improvement"
    if value >= NUMERIC_BENCHMARKS[metric_name]["high"]:
        return "strong"
    return "moderate"


def build_student_insights(questions_answered, active_participation, unit_test_percent, periodic_test_percent, raised, resources, announcements, discussion, absence, parent_answering, parent_satisfaction):
    strengths = []
    focus_areas = []
    suggestions = []
    metric_rows = []

    factor_config = [
        ("QuestionsAnswered", questions_answered, "Question answering and doubt-solving", "Answer or ask more questions in class to improve clarity and confidence."),
        ("ActiveParticipation", active_participation, "Active participation", "Participate consistently in class activities, oral responses, and collaborative tasks."),
        ("UnitTestPercent", unit_test_percent, "Unit test performance", "Review weak chapters from the latest unit test and practice short questions daily."),
        ("PeriodicTestPercent", periodic_test_percent, "Periodic test performance", "Build a weekly revision plan and solve previous periodic-test style questions."),
        ("raisedhands", raised, "Class participation", "Answer at least one question or contribute once in each class session."),
        ("VisITedResources", resources, "Learning resource usage", "Spend regular time with digital notes, reference videos, and practice material."),
        ("AnnouncementsView", announcements, "Academic follow-up", "Check notices and academic updates daily so deadlines are not missed."),
        ("Discussion", discussion, "Discussion participation", "Ask doubts early and take part in peer discussion or group study."),
    ]

    for key, value, label, suggestion in factor_config:
        status = factor_status(value, key)
        metric_rows.append({
            "Factor": label,
            "Current score": int(value),
            "Status": "Strong" if status == "strong" else "Needs improvement" if status == "needs_improvement" else "Moderate",
        })
        if status == "strong":
            strengths.append(label)
        elif status == "needs_improvement":
            focus_areas.append(label)
            suggestions.append(suggestion)

    if absence == "Above-7":
        focus_areas.append("Attendance consistency")
        suggestions.append("Reduce absence frequency and track weekly attendance with a simple follow-up plan.")
        attendance_score = 0
    else:
        strengths.append("Attendance consistency")
        attendance_score = 100

    metric_rows.append({
        "Factor": "Attendance consistency",
        "Current score": 100 if absence == "Under-7" else 0,
        "Status": "Strong" if absence == "Under-7" else "Needs improvement",
    })

    if parent_answering == "No":
        focus_areas.append("Home academic follow-up")
        suggestions.append("Increase parent or guardian follow-up on weekly progress and assignments.")
        parent_answering_score = 0
    else:
        strengths.append("Home academic follow-up")
        parent_answering_score = 100

    metric_rows.append({
        "Factor": "Home academic follow-up",
        "Current score": 100 if parent_answering == "Yes" else 0,
        "Status": "Strong" if parent_answering == "Yes" else "Needs improvement",
    })

    if parent_satisfaction == "Bad":
        focus_areas.append("School satisfaction and support")
        suggestions.append("Review classroom support, teacher feedback, and learning comfort with the student.")
        satisfaction_score = 0
    else:
        strengths.append("School satisfaction and support")
        satisfaction_score = 100

    metric_rows.append({
        "Factor": "School satisfaction and support",
        "Current score": 100 if parent_satisfaction == "Good" else 0,
        "Status": "Strong" if parent_satisfaction == "Good" else "Needs improvement",
    })

    weighted_scores = [
        questions_answered * 10,
        active_participation,
        unit_test_percent,
        periodic_test_percent,
        raised,
        resources,
        announcements,
        discussion,
        attendance_score,
        parent_answering_score,
        satisfaction_score,
    ]
    improvement_score = round(sum(weighted_scores) / len(weighted_scores), 1)

    if improvement_score >= 75:
        improvement_band = "Low risk"
    elif improvement_score >= 50:
        improvement_band = "Moderate risk"
    else:
        improvement_band = "High support needed"

    if not focus_areas:
        suggestions.append("Maintain the current routine and keep consistency across attendance, engagement, and follow-up.")

    return {
        "strengths": list(dict.fromkeys(strengths)),
        "focus_areas": list(dict.fromkeys(focus_areas)),
        "suggestions": list(dict.fromkeys(suggestions)),
        "metric_rows": pd.DataFrame(metric_rows),
        "improvement_score": improvement_score,
        "improvement_band": improvement_band,
    }

with st.container(border=True):
    st.subheader("Dataset Preview")
    if "latest_submission_message" in st.session_state:
        st.success(st.session_state.pop("latest_submission_message"))
    if data.empty:
        st.info("No active rows are currently available in the dataset. Submit the form below to start collecting active records.")
    else:
        st.dataframe(data.head(), width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Add New Data")

    st.markdown("""
Add a new student record here and save it directly into the dataset file used by this dashboard.
""")

    category_columns = [
        "gender",
        "StageID",
        "Topic",
        "ParentAnsweringSurvey",
        "ParentschoolSatisfaction",
        "StudentAbsenceDays",
        "Class",
    ]
    category_options = {
        col: sorted(data[col].dropna().astype(str).unique().tolist()) if not data.empty else []
        for col in category_columns
    }
    for col in category_columns:
        category_options[col] = merged_category_options(
            category_options[col],
            DEFAULT_CATEGORY_OPTIONS[col],
        )

    with st.form("add_student_record_form", clear_on_submit=True):
        form_col1, form_col2, form_col3 = st.columns(3)

        with form_col1:
            student_name = st.text_input("Student Name", key="student_name_input")
            gender = st.selectbox("Gender", category_options["gender"], key="gender_input")
            stage = st.selectbox("Stage", category_options["StageID"], key="stage_input")
            topic = st.selectbox("Topic", category_options["Topic"], key="topic_input")
            questions_answered = st.number_input("Questions Answered", min_value=0, max_value=10, value=0, step=1, key="questions_answered_input")

        with form_col2:
            active_participation = st.number_input("Active Participation", min_value=0, max_value=100, value=0, step=1, key="active_participation_input")
            unit_test_marks = st.number_input("Unit Test Marks", min_value=0, max_value=100, value=0, step=1, key="unit_test_marks_input")
            unit_test_max_marks = st.number_input("Unit Test Max Marks", min_value=1, max_value=100, value=100, step=1, key="unit_test_max_marks_input")
            periodic_test_marks = st.number_input("Periodic Test Marks", min_value=0, max_value=100, value=0, step=1, key="periodic_test_marks_input")
            raisedhands_value = st.number_input("Raised Hands", min_value=0, max_value=10, value=0, step=1, key="raisedhands_input")

        with form_col3:
            periodic_test_max_marks = st.number_input("Periodic Test Max Marks", min_value=1, max_value=100, value=100, step=1, key="periodic_test_max_marks_input")
            discussion = st.number_input("Discussion", min_value=0, max_value=10, value=0, step=1, key="discussion_input")
            parent_answering = st.selectbox("Parent Answering Survey", category_options["ParentAnsweringSurvey"], key="parent_answering_input")
            parent_satisfaction = st.selectbox("Parent School Satisfaction", category_options["ParentschoolSatisfaction"], key="parent_satisfaction_input")
            absence_days = st.selectbox("Student Absence Days", category_options["StudentAbsenceDays"], key="absence_days_input")
            student_class = st.selectbox("Class", category_options["Class"], key="student_class_input")
            submit_new_record = st.form_submit_button("Save New Record", use_container_width=True)

        if submit_new_record:
            questions_answered = max(0, min(int(questions_answered), 10))
            raisedhands_value = max(0, min(int(raisedhands_value), 10))
            discussion = max(0, min(int(discussion), 10))
            visited_resources = max(0, min(int(DEFAULT_VISITED_RESOURCES), 100))
            announcements_view = max(0, min(int(DEFAULT_ANNOUNCEMENTS_VIEW), 10))
            new_row = {
                "StudentName": student_name.strip() if student_name.strip() else "Unnamed Student",
                "gender": gender,
                "StageID": stage,
                "SectionID": "A",
                "Topic": topic,
                "QuestionsAnswered": questions_answered,
                "ActiveParticipation": int(active_participation),
                "UnitTestMarks": int(unit_test_marks),
                "UnitTestMaxMarks": int(unit_test_max_marks),
                "PeriodicTestMarks": int(periodic_test_marks),
                "PeriodicTestMaxMarks": int(periodic_test_max_marks),
                "raisedhands": raisedhands_value,
                "VisITedResources": int(visited_resources),
                "AnnouncementsView": announcements_view,
                "Discussion": discussion,
                "ParentAnsweringSurvey": parent_answering,
                "ParentschoolSatisfaction": parent_satisfaction,
                "StudentAbsenceDays": absence_days,
                "Class": student_class,
            }

            pd.DataFrame([new_row], columns=CSV_COLUMNS).to_csv(
                csv_path,
                mode="a",
                header=False,
                index=False,
            )
            st.session_state["latest_submission"] = new_row
            st.session_state["latest_submission_message"] = "New student record added successfully."
            st.rerun()

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

    if data.empty:
        st.info("Correlation heatmap will appear after active records are available.")
    else:
        summary_means = pd.DataFrame({
            "Factor": [
                "Questions Answered",
                "Active Participation",
                "Unit Test Percentage",
                "Periodic Test Percentage",
                "Raised Hands",
                "Visited Resources",
                "Announcements View",
                "Discussion",
            ],
            "Average Score": [
                data["QuestionsAnswered"].mean() * 10,
                data["ActiveParticipation"].mean(),
                data["UnitTestPercent"].mean(),
                data["PeriodicTestPercent"].mean(),
                data["raisedhands"].mean(),
                data["VisITedResources"].mean(),
                data["AnnouncementsView"].mean(),
                data["Discussion"].mean(),
            ],
        })
        fig, ax = plt.subplots(figsize=(9, 5))
        sb.barplot(data=summary_means, x="Average Score", y="Factor", ax=ax, color="#8ecae6")
        ax.set_title("Average student factor scores")
        ax.set_xlabel("Average score")
        ax.set_ylabel("Factor")
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

    if data.empty or data["Class"].nunique() < 2:
        st.info("Feature relationship view needs active records from at least two performance classes.")
    else:
        class_summary = data.groupby("Class")[[
            "ActiveParticipation",
            "UnitTestPercent",
            "PeriodicTestPercent",
            "raisedhands",
            "VisITedResources",
        ]].mean().reset_index()
        class_summary = class_summary.melt(id_vars="Class", var_name="Factor", value_name="Average Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sb.barplot(data=class_summary, x="Factor", y="Average Score", hue="Class", ax=ax)
        ax.set_title("Average factor score by performance class")
        ax.set_xlabel("Factor")
        ax.set_ylabel("Average score")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Explore Data")
    if data.empty:
        st.info("Exploration charts will appear after active records are added.")
    else:
        option = st.selectbox("Choose what you want to see", [
            "Distribution of Study Activity",
            "Performance vs Activity",
            "Class Distribution",
        ])

        if option == "Distribution of Study Activity":
            st.markdown("Shows how active students are overall")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sb.histplot(data["ActiveParticipation"], kde=True, ax=ax)
            ax.set_title("How often students are actively involved")
            ax.set_xlabel("Active participation")
            ax.set_ylabel("Number of students")
            st.pyplot(fig, width="stretch")

        elif option == "Performance vs Activity":
            st.markdown("Compare activity levels across performance groups")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            performance_activity = data.groupby("Class")[["ActiveParticipation", "UnitTestPercent", "PeriodicTestPercent"]].mean().reset_index()
            performance_activity = performance_activity.melt(id_vars="Class", var_name="Factor", value_name="Average Score")
            sb.barplot(data=performance_activity, x="Factor", y="Average Score", hue="Class", ax=ax)
            ax.set_title("Performance compared with key student factors")
            ax.set_xlabel("Class")
            ax.set_ylabel("Average score")
            plt.xticks(rotation=20, ha="right")
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
    st.subheader("Learning Overview")

    st.markdown("""
These visuals help explain the current student records in a more direct and easy-to-read way.
""")

model_features = [
    "QuestionsAnswered",
    "ActiveParticipation",
    "UnitTestPercent",
    "PeriodicTestPercent",
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
    "StudentAbsenceDays",
]

X = pd.DataFrame()
y = pd.Series(dtype=str)
scaler = None
model_dt = None
model_rf = None
model_lr = None
pred_dt = pred_rf = pred_lr = None
acc_dt = acc_rf = acc_lr = None
y_test = None

if model_ready:
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
    if data.empty:
        st.info("Student overview visuals will appear after active records are added.")
    else:
        overview_cols = st.columns(4)
        overview_cols[0].metric("Active students", len(data))
        overview_cols[1].metric("Average unit test %", f"{data['UnitTestPercent'].mean():.1f}")
        overview_cols[2].metric("Average periodic test %", f"{data['PeriodicTestPercent'].mean():.1f}")
        overview_cols[3].metric("Average participation", f"{data['ActiveParticipation'].mean():.1f}")

        easy_visual = pd.DataFrame({
            "Area": [
                "Unit Test %",
                "Periodic Test %",
                "Active Participation",
                "Resource Usage",
                "Discussion",
                "Attendance Support",
            ],
            "Average Score": [
                data["UnitTestPercent"].mean(),
                data["PeriodicTestPercent"].mean(),
                data["ActiveParticipation"].mean(),
                data["VisITedResources"].mean(),
                data["Discussion"].mean(),
                100 * (data["StudentAbsenceDays"] == "Under-7").mean(),
            ],
        })
        fig, ax = plt.subplots(figsize=(9, 5))
        sb.barplot(data=easy_visual, x="Area", y="Average Score", ax=ax, color="#90be6d")
        ax.set_title("Overall student support overview")
        ax.set_xlabel("Area")
        ax.set_ylabel("Average score")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")

st.divider()
with st.container(border=True):
    st.subheader("Try Your Own Prediction")

    st.markdown("""
Prediction now runs from the data submission form above.

- Submit a student record once
- The record is saved into the CSV file
- The prediction response and improvement report appear here
""")

    submitted_row = st.session_state.get("latest_submission")

    if not submitted_row:
        st.info("Submit the Add New Data form to save a student record and generate the response here.")
    else:
        student_name = submitted_row["StudentName"]
        questions_answered = submitted_row["QuestionsAnswered"]
        active_participation = submitted_row["ActiveParticipation"]
        unit_test_marks = submitted_row["UnitTestMarks"]
        unit_test_max_marks = submitted_row["UnitTestMaxMarks"]
        periodic_test_marks = submitted_row["PeriodicTestMarks"]
        periodic_test_max_marks = submitted_row["PeriodicTestMaxMarks"]
        unit_test_percent = round((unit_test_marks / max(unit_test_max_marks, 1)) * 100, 1)
        periodic_test_percent = round((periodic_test_marks / max(periodic_test_max_marks, 1)) * 100, 1)
        raised = submitted_row["raisedhands"]
        resources = submitted_row["VisITedResources"]
        announcements = submitted_row["AnnouncementsView"]
        discussion = submitted_row["Discussion"]
        absence = submitted_row["StudentAbsenceDays"]
        parent_answering_prediction = submitted_row["ParentAnsweringSurvey"]
        parent_satisfaction_prediction = submitted_row["ParentschoolSatisfaction"]

        insight_bundle = build_student_insights(
            questions_answered,
            active_participation,
            unit_test_percent,
            periodic_test_percent,
            raised,
            resources,
            announcements,
            discussion,
            absence,
            parent_answering_prediction,
            parent_satisfaction_prediction,
        )

        predicted_label = "Model not ready yet"
        probability_df = pd.DataFrame()

        if model_ready:
            X_latest = pd.get_dummies(data[model_features], columns=["StudentAbsenceDays"], dtype=float)
            input_data = pd.DataFrame([{
                "QuestionsAnswered": questions_answered,
                "ActiveParticipation": active_participation,
                "UnitTestPercent": unit_test_percent,
                "PeriodicTestPercent": periodic_test_percent,
                "raisedhands": raised,
                "VisITedResources": resources,
                "AnnouncementsView": announcements,
                "Discussion": discussion,
                f"StudentAbsenceDays_{absence}": 1.0,
            }])
            input_data = input_data.reindex(columns=X_latest.columns, fill_value=0.0)
            input_scaled = scaler.transform(input_data)

            prediction = model_rf.predict(input_scaled)
            prediction_probabilities = model_rf.predict_proba(input_scaled)[0]
            probability_lookup = {
                label: float(probability)
                for label, probability in zip(model_rf.classes_, prediction_probabilities)
            }
            predicted_label = prediction[0]
            probability_df = pd.DataFrame({
                "Class": model_rf.classes_,
                "Probability": [round(probability_lookup[label], 3) for label in model_rf.classes_],
            }).sort_values(by="Probability", ascending=False)
        else:
            predicted_label = score_to_prediction_label(insight_bundle["improvement_score"])

        report_lines = [
            "Student Improvement Report",
            "",
            f"Student Name: {student_name}",
            f"Predicted Performance: {predicted_label}",
            f"Improvement Risk Level: {insight_bundle['improvement_band']}",
            f"Improvement Score: {insight_bundle['improvement_score']} / 100",
            "",
            "Strengths:",
        ]
        if insight_bundle["strengths"]:
            report_lines.extend([f"- {item}" for item in insight_bundle["strengths"]])
        else:
            report_lines.append("- No clear strengths identified yet")

        report_lines.extend(["", "Focus Areas:"])
        if insight_bundle["focus_areas"]:
            report_lines.extend([f"- {item}" for item in insight_bundle["focus_areas"]])
        else:
            report_lines.append("- No urgent focus areas identified")

        report_lines.extend(["", "Recommended Actions:"])
        report_lines.extend([f"- {item}" for item in insight_bundle["suggestions"]])
        report_text = "\n".join(report_lines)

        input_col, result_col = st.columns([1.05, 0.95])

        with input_col:
            submitted_summary = pd.DataFrame([
                {"Field": "Student Name", "Value": student_name},
                {"Field": "Questions Answered", "Value": questions_answered},
                {"Field": "Active Participation", "Value": active_participation},
                {"Field": "Unit Test Marks", "Value": f"{unit_test_marks} / {unit_test_max_marks} ({unit_test_percent}%)"},
                {"Field": "Periodic Test Marks", "Value": f"{periodic_test_marks} / {periodic_test_max_marks} ({periodic_test_percent}%)"},
                {"Field": "Raised Hands", "Value": raised},
                {"Field": "Discussion", "Value": discussion},
                {"Field": "Absence", "Value": absence},
                {"Field": "Parent Answering Survey", "Value": parent_answering_prediction},
                {"Field": "Parent School Satisfaction", "Value": parent_satisfaction_prediction},
                {"Field": "Class saved in CSV", "Value": submitted_row["Class"]},
            ])
            st.dataframe(submitted_summary, hide_index=True, width="stretch")

        with result_col:
            st.success(f"Predicted Performance: {predicted_label}")
            if model_ready:
                st.dataframe(probability_df, hide_index=True, width="stretch")
            else:
                st.info("Prediction is currently based on the student improvement score until enough active records are available for ML training.")

            st.info(f"Improvement Risk Level: {insight_bundle['improvement_band']}")
            st.metric("Improvement Score", f"{insight_bundle['improvement_score']} / 100")

            st.markdown("""
### Simple Insight:
- Stay active in class
- Use learning resources
- Avoid too many absences

These greatly improve performance.
""")

        st.markdown("### Student Snapshot")
        snapshot_cols = st.columns(4)
        snapshot_cols[0].metric("Unit Test %", f"{unit_test_percent}")
        snapshot_cols[1].metric("Periodic Test %", f"{periodic_test_percent}")
        snapshot_cols[2].metric("Active Participation", f"{active_participation}")
        snapshot_cols[3].metric("Questions Answered", f"{questions_answered}")

        st.markdown("### Student vs Healthy Benchmark")
        comparison_df = pd.DataFrame({
            "Factor": [
                "Unit Test %",
                "Periodic Test %",
                "Active Participation",
                "Raised Hands",
                "Resources Used",
                "Discussion",
            ],
            "Student Score": [
                unit_test_percent,
                periodic_test_percent,
                active_participation,
                raised,
                resources,
                discussion,
            ],
            "Healthy Target": [
                max(70, NUMERIC_BENCHMARKS["UnitTestPercent"]["high"]),
                max(70, NUMERIC_BENCHMARKS["PeriodicTestPercent"]["high"]),
                max(70, NUMERIC_BENCHMARKS["ActiveParticipation"]["high"]),
                max(70, NUMERIC_BENCHMARKS["raisedhands"]["high"]),
                max(70, NUMERIC_BENCHMARKS["VisITedResources"]["high"]),
                max(70, NUMERIC_BENCHMARKS["Discussion"]["high"]),
            ],
        })
        comparison_melted = comparison_df.melt(id_vars="Factor", var_name="Type", value_name="Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sb.barplot(data=comparison_melted, x="Factor", y="Score", hue="Type", ax=ax)
        ax.set_title("Student performance compared with healthy target levels")
        ax.set_xlabel("Factor")
        ax.set_ylabel("Score")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")

        st.markdown("### Weak Factors Detected")
        if insight_bundle["focus_areas"]:
            for item in insight_bundle["focus_areas"]:
                st.warning(item)
        else:
            st.success("No major weak factor is currently flagged from the entered profile.")

        st.markdown("### Student Strengths")
        if insight_bundle["strengths"]:
            for item in insight_bundle["strengths"]:
                st.success(item)
        else:
            st.info("Strengths will appear here once the profile has stronger engagement signals.")

        st.markdown("### Improvement Suggestions")
        for item in insight_bundle["suggestions"]:
            st.write(f"- {item}")

        st.markdown("### Student Factor Summary")
        st.dataframe(insight_bundle["metric_rows"], hide_index=True, width="stretch")
        st.download_button(
            "Download Student Report",
            data=report_text,
            file_name="student_improvement_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
