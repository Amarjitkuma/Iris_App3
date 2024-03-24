#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train_scaled, y_train)

# Streamlit app
st.title("Iris Classification")

# User input
sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Predict function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    scaled_input = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = mlp_classifier.predict(scaled_input)
    return prediction

# Display prediction
if st.button("Predict"):
    prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    predicted_class = iris.target_names[prediction[0]]
    st.write(f"Predicted Iris Species: {predicted_class}")

# Show Iris dataset information
st.write("Iris dataset classes:", iris.target_names)


# In[ ]:




