

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# Function to load data
@st.cache
def load_data():
    return pd.read_csv('tweets.csv')

# Function to get a deep copy of the data
def get_data_copy():
    return deepcopy(load_data())

# Plotting function
def plot_data(data, plot_type, column=None):
    if plot_type == 'Histogram':
        fig, ax = plt.subplots()
        data[column].hist(ax=ax)
        st.pyplot(fig)
    elif plot_type == 'Scatter Plot':
        if len(data.columns) > 1:
            y_column = st.selectbox("Select another column for Y-axis", [col for col in data.columns if col != column])
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[column], y=data[y_column], data=data, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Not enough columns for a scatter plot!")
    elif plot_type == 'Heatmap':
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, ax=ax)
        st.pyplot(fig)
    elif plot_type == 'Bar Plot':
        fig, ax = plt.subplots()
        data[column].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    elif plot_type == 'Pie Chart':
        fig, ax = plt.subplots()
        data[column].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        st.pyplot(fig)
    elif plot_type == 'Density Plot':
        fig, ax = plt.subplots()
        data[column].plot(kind='density', ax=ax)
        st.pyplot(fig)

# Function to preprocess text data
def preprocess_text(data):
    vectorizer = CountVectorizer(stop_words='english')
    features = vectorizer.fit_transform(data['tweet']).toarray()
    return features, data['sentiment'], vectorizer

# Main app
def main():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Navigate", options=["Home", "Data Analysis", "Data Cleaning", "Train Model", "Prediction"])

    if menu == "Home":
        st.title('Sentiment Analysis Application')
        st.image("image.png", use_column_width=True)
        st.write("Welcome to the Sentiment Analysis Application using a Decision Tree model. Navigate through the sidebar to explore various functionalities.")

    if menu == "Data Analysis":
        st.title('Data Analysis')
        data = get_data_copy()
        st.write(data.head())
        st.write(data.tail())
        st.subheader('Dataset Information')
        st.write("Shape of the dataset:", data.shape)
        st.write("Missing Values:")
        st.write(data.isnull().sum())
        st.write("Statistical Description:")
        st.write(data.describe())
        plot_type = st.selectbox("Select analysis type", ['Histogram', 'Bar Plot', 'Pie Chart'])
        if plot_type not in ['Heatmap', 'Pie Chart', 'Density Plot']:
            column = st.selectbox("Select the column for plotting", data.columns)
            plot_data(data, plot_type, column)
        else:
            plot_data(data, plot_type, data.columns[0])

    if menu == "Data Cleaning":
        st.title('Data Cleaning')
        data = get_data_copy()
        st.write("Original Data", data.head())
        data['tweet'] = data['tweet'].str.replace('[^\w\s]', '', regex=True).str.lower()
        st.write("Cleaned Data", data.head())

    if menu == "Train Model":
        st.title('Train Model')
        data = get_data_copy()
        X, y, vectorizer = preprocess_text(data)
        y = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        st.subheader('Model Evaluation Metrics')
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Training Accuracy: {train_accuracy:.2f}')
        st.write(f'Testing Accuracy: {test_accuracy:.2f}')
        report  = classification_report(y_test,y_pred)
        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix and other metrics
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        # ROC and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="AUC = {:.3f}".format(auc))
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Decision Tree Plot
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(model, filled=True, ax=ax, feature_names=vectorizer.get_feature_names_out())
        st.pyplot(fig)

    if menu == "Prediction":
        st.title('Prediction')
        data = get_data_copy()
        _, _, vectorizer = preprocess_text(data)
        user_input = st.text_input("Enter text to predict sentiment:")
        if user_input:
            input_features = vectorizer.transform([user_input]).toarray()
            data = get_data_copy()
            X, y, vectorizer = preprocess_text(data)
            y = LabelEncoder().fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            prediction = model.predict(input_features)
            probability = model.predict_proba(input_features)[0]
            st.write(f'Predicted Sentiment: {prediction[0]}')
            fig, ax = plt.subplots()
            ax.bar(['Negative', 'Positive'], probability)  # Adjust labels as per your classes
            ax.set_ylabel('Probability')
            ax.set_title('Probability Plot')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
