import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction using Decision Tree")

# خواندن داده
data = pd.read_csv("housepricedata.csv")

st.subheader("Dataset")
st.dataframe(data.head())

X = data.drop("AboveMedianPrice", axis=1)
y = data["AboveMedianPrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)

st.subheader("Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1 Score", f"{f1:.3f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, prediction))

st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(
    confusion_matrix(y_test, prediction),
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)
st.pyplot(fig)

st.subheader("Decision Tree")

fig2, ax2 = plt.subplots(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Below Median","Above Median"],
    filled=True,
    ax=ax2
)
st.pyplot(fig2)

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance")

fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.barh(importance["Feature"], importance["Importance"])
st.pyplot(fig3)

result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": prediction
})

st.subheader("Prediction Result")
st.dataframe(result)

csv = result.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download prediction.csv",
    csv,
    file_name="prediction.csv",
    mime="text/csv"
)