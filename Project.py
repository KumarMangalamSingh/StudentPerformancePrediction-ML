import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w
w.filterwarnings('ignore')

data = pd.read_csv("AI-Data.csv")

# ---- Correlation Heatmap ----
plt.figure(figsize=(12, 8))
sb.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

ch = 0
while(ch != 10):
    print("1.Marks Class Count Graph\t2.Marks Class Semester-wise Graph\n3.Marks Class Gender-wise Graph\t4.Marks Class Nationality-wise Graph\n5.Marks Class Grade-wise Graph\t6.Marks Class Section-wise Graph\n7.Marks Class Topic-wise Graph\t8.Marks Class Stage-wise Graph\n9.Marks Class Absent Days-wise\t10.No Graph\n")
    ch = int(input("Enter Choice: "))
    
    if (ch == 1):
        sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 2):
        sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 3):
        sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 4):
        sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 5):
        sb.countplot(x='GradeID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 6):
        sb.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 7):
        sb.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 8):
        sb.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

    elif (ch == 9):
        sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.show()

if(ch == 10):
    print("Exiting..")

# Drop columns
data = data.drop("gender", axis=1)
data = data.drop("StageID", axis=1)
data = data.drop("GradeID", axis=1)
data = data.drop("NationalITy", axis=1)
data = data.drop("PlaceofBirth", axis=1)
data = data.drop("SectionID", axis=1)
data = data.drop("Topic", axis=1)
data = data.drop("Semester", axis=1)
data = data.drop("Relation", axis=1)
data = data.drop("ParentschoolSatisfaction", axis=1)
data = data.drop("ParentAnsweringSurvey", axis=1)
data = data.drop("AnnouncementsView", axis=1)

# ✅ FIX 1: Proper shuffle
data = u.shuffle(data)

# ✅ FIX 2: SAFE encoding (prevents 'M' error completely)
data = pd.get_dummies(data)

# Split data
ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls = data.values[:, 4]

feats_Train = feats[0:ind]
feats_Test = feats[ind:len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[ind:len(lbls)]

# Models
modelD = tr.DecisionTreeClassifier()
modelD.fit(feats_Train, lbls_Train)
lbls_predD = modelD.predict(feats_Test)
print("\nDecision Tree Accuracy:", m.accuracy_score(lbls_Test, lbls_predD))

modelR = es.RandomForestClassifier()
modelR.fit(feats_Train, lbls_Train)
lbls_predR = modelR.predict(feats_Test)
print("Random Forest Accuracy:", m.accuracy_score(lbls_Test, lbls_predR))

modelP = lm.Perceptron()
modelP.fit(feats_Train, lbls_Train)
lbls_predP = modelP.predict(feats_Test)
print("Perceptron Accuracy:", m.accuracy_score(lbls_Test, lbls_predP))

modelL = lm.LogisticRegression(max_iter=1000)
modelL.fit(feats_Train, lbls_Train)
lbls_predL = modelL.predict(feats_Test)
print("Logistic Regression Accuracy:", m.accuracy_score(lbls_Test, lbls_predL))

modelN = nn.MLPClassifier(activation="logistic")
modelN.fit(feats_Train, lbls_Train)
lbls_predN = modelN.predict(feats_Test)
print("MLP Accuracy:", m.accuracy_score(lbls_Test, lbls_predN))