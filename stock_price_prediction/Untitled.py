import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import plotly.graph_objects as go


# Set page configuration before any other Streamlit component
st.set_page_config(layout="wide", page_title="Stock Prediction App")

# Sidebar for options
st.sidebar.header("⚙️ Options")
st.sidebar.subheader("📊 Stock Data")

# Stock selection
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., ^NSEI)", "^NSEI")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-09-11"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-09-11"))
classifier = st.sidebar.radio("Select Classifier", ("Naive Bayes", "ID3 Decision Tree"))

# Download stock data
if start_date != end_date:
    stock_data = yf.download(stock, start=start_date, end=end_date)
    if not stock_data.empty:
        st.sidebar.success(f"Downloaded data for {stock} from {start_date} to {end_date}")
    else:
        st.sidebar.error("Failed to download stock data. Please check the ticker and dates.")
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])
    fig.update_layout(title=f'Candlestick chart for {stock}',
                      yaxis_title='Stock Price',
                      xaxis_title='Date',
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)
# Display stock data
if 'stock_data' in locals() and not stock_data.empty:
    st.write("## Stock Data", stock_data)

    # Prepare data for classification
    stock_data['next_close'] = stock_data['Close'].shift(-1)
    stock_data['close_class'] = np.where(stock_data["Close"] < stock_data['next_close'], 1, 0)
    stock_data.dropna(inplace=True)

    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_data['close_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Classifier selection

    # st.sidebar.selectbox("Select Classifier", ("Naive Bayes", "ID3 Decision Tree"))

    if classifier == "Naive Bayes":
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)

        # Display accuracy and confusion matrix
        st.write("### Naive Bayes Classifier")
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}%")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Price Down', 'Price Up'], yticklabels=['Price Down', 'Price Up'], ax=ax)
        st.pyplot(fig)

    elif classifier == "ID3 Decision Tree":
        id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
        id3_model.fit(X_train, y_train)
        y_pred = id3_model.predict(X_test)

        # Display accuracy and decision tree
        st.write("### ID3 Decision Tree Classifier")
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy*100:.2f}%")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(id3_model, filled=True, feature_names=X.columns, class_names=["0", "1"], rounded=True, ax=ax)
        st.pyplot(fig)
