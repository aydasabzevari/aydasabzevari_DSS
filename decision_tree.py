
```python
# ============================================
# Project: House Price Prediction using Decision Tree
# Course: Data Mining
# ============================================

# وارد کردن کتابخانه‌ها
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns

# ============================================
# خواندن دیتاست
# ============================================

data = pd.read_csv("housepricedata.csv")

# نمایش پنج سطر اول
print(data.head())

# ============================================
# جدا کردن ویژگی‌ها و ستون هدف
# ============================================

X = data.drop("AboveMedianPrice", axis=1)

y = data["AboveMedianPrice"]

# ============================================
# تقسیم داده‌ها
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)

# ============================================
# ساخت مدل
# ============================================

model = DecisionTreeClassifier(random_state=42)

# آموزش مدل

model.fit(X_train, y_train)

# ============================================
# پیش‌بینی
# ============================================

prediction = model.predict(X_test)

# ============================================
# محاسبه معیارها
# ============================================

accuracy = accuracy_score(y_test, prediction)

precision = precision_score(y_test, prediction)

recall = recall_score(y_test, prediction)

f1 = f1_score(y_test, prediction)

print()

print("Accuracy =", accuracy)

print("Precision =", precision)

print("Recall =", recall)

print("F1 Score =", f1)

# ============================================
# گزارش طبقه‌بندی
# ============================================

print()

print(classification_report(y_test, prediction))

# ============================================
# ماتریس درهم‌ریختگی
# ============================================

cm = confusion_matrix(y_test, prediction)

plt.figure(figsize=(6,5))

sns.heatmap(

cm,

annot=True,

fmt="d",

cmap="Blues"

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ============================================
# رسم درخت تصمیم
# ============================================

plt.figure(figsize=(20,10))

plot_tree(

model,

feature_names=X.columns,

class_names=["Below Median","Above Median"],

filled=True

)

plt.show()

# ============================================
# اهمیت ویژگی‌ها
# ============================================

importance = model.feature_importances_

plt.figure(figsize=(10,8))

plt.barh(

X.columns,

importance

)

plt.title("Feature Importance")

plt.xlabel("Importance")

plt.ylabel("Features")

plt.show()

# ============================================
# ذخیره مدل
# ============================================

import joblib

joblib.dump(

model,

"decision_tree_model.pkl"

)

print()

print("Model Saved Successfully")

# ============================================
# ذخیره پیش‌بینی‌ها
# ============================================

result = pd.DataFrame({

"Actual": y_test,

"Predicted": prediction

})

result.to_csv(

"prediction.csv",

index=False

)

print("Prediction File Saved")
```
