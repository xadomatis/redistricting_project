import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from plotnine import *

st.title("ML application!")

st.write("""
# Explore the data
### Comparing classifiers""")

# st.selectbox("Choose data:", ("Iris", "Breast Cancer")) 

# Make selectbox a sidebar
dataset_name = st.sidebar.selectbox("Choose data:", ("Iris", "Breast Cancer", "Wine dataset")) 


## Classifiers
c_name = st.sidebar.selectbox("Choose an algorithm:", ("Logistic Regression", "SVM", "Linear SVM"))

def get_dataset(dataset_name):
     if dataset_name == "Iris":
         data = datasets.load_iris()
     elif dataset_name == "Breast Cancer":
         data = datasets.load_breast_cancer()
     else:
         data = datasets.load_wine()
     X = data.data
     y = data.target
     return X, y

X, y = get_dataset(dataset_name)
st.write("Obs:", X.shape[0])
st.write("Features:", X.shape[1])
st.write("Number of classes", len(np.unique(y)))

#### Add parameters
def add_param(c_name):
    params = dict()
    if c_name == "Logistic Regression":
        C = st.sidebar.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    elif c_name == "SVM":
        C = st.sidebar.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    else:
        C = st.sidebar.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    return params

#### Call add_param function with classifier name as argument (selected above)
params = add_param(c_name)

#### Add classifier: takes name of classifier we want to use and parameters for the one

def add_class(c_name, params):
    if c_name == "Logistic Regression":
        mod = LogisticRegression(C=params["C"])
    elif c_name == "SVM":
        mod = SVC(C=params["C"])
    else:
        mod = LinearSVC(C=params["C"])
    return mod

#### Call add class function to add classifier and associated parameters
mod = add_class(c_name, params)


### Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

mod.fit(X_train, y_train)
y_pred = mod.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Algorithm = {c_name}")
st.write(f"Accuracy (same as test) = {round(acc, 2)}")
st.write(f"Training score: {round(mod.score(X_train, y_train), 2)}")
st.write(f"Test score: {round(mod.score(X_test, y_test), 2)}")


#### Plot
import pandas as pd
import altair as alt

### Apply PCA to get 2 dimensions
pca = PCA(2)
X_plot = pd.DataFrame(pca.fit_transform(X))

X_plot.columns = ["x", "y"]

alt_plot = alt.Chart(X_plot).mark_circle().encode(x="x",
y="y").interactive()
alt_plot
