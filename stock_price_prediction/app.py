import streamlit as st
from iD3_classifier import StockAnalysisid3
from Naive_Bayes_classifier import NaiveBayesClassifier
from linear_Regression import StockPredictor
from linear_reg import StockPricePredictor
from GBoost import StockAnalysisGb
from randomforest import StockAnalysisrf
import datetime

# Set page configuration before any other Streamlit component
st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")

# Injecting custom CSS styles
st.markdown("""
    <style>
    /* Background and text color */
    body {

        background-color: #1E1E1E;
        color: #F0F0F0;

    }

    /* Title and header styling */
    .css-1aumxhk {
        color: ##FF0000 !important;

    }

    h1, h2, h3 {
        color: #FF0000;
        font-family: 'Verdana', sans-serif;
        font-weight: bold;
    }
    h2 {
        font-size: 30px
    }
    h3 {
        font-size: 20px
    }

    /* Sidebar style */
    .css-1d391kg {
        background-color: #2C2C2C !important;
        color: #F0F0F0;
    }

    /* Dropdown styling */
    .stSelectbox {
        background-color: #333333 !important;
        color: #FFFFFF !important;
    }

    /* Radio button and select styling */
    .stRadio > div {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 10px;
    }

    /* Data plot area */
    .stPlotlyChart {
        background-color: #262626 !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #00FFAB !important;
        color: black;
        border-radius: 12px;
        padding: 8px 16px;
    }

    /* Sidebar button styling */
    .stFormSubmitButton > button {
        background-color: #00FFAB !important;
        color: black;
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 12px;
    }

    ::-webkit-scrollbar-track {
        background: #2C2C2C;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #00FFAB;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Set the title for the application
st.title("📈 Stock Analysis Dashboard")

# Sidebar with a dropdown menu for selecting stocks
st.sidebar.header("⚙️ Options")
st.sidebar.subheader("📊 Stock Selection")

stock_option = st.sidebar.selectbox(
    'Choose a Stock:',
    ['NIFTY', 'VBL', 'CoalIndia', 'ITC','Havells'],
    index=0
)

# Display the selected stock option in the main section
if stock_option:
    st.header(f"📊 Stock Selected: {stock_option}")

    # Map the selected stock to its symbol
    stock_map = {
        'NIFTY': "^NSEI",
        'VBL': "VBL.NS",
        'CoalIndia': "COALINDIA.NS",
        'ITC': "ITC.NS",
        'Havells': "HAVELLS.NS"
    }
    stock = stock_map.get(stock_option)
    st.write("### Candlestick chart")

    # Sidebar date input for selecting date range
    st.sidebar.subheader("📅 Date Range")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    try:
        predictor = StockPricePredictor(stock, start_date, end_date)
        predictor.candlestick_plot()
    except:
        st.write("No stock selected")
    if start_date and end_date and start_date != end_date:
        st.write(f"Analyzing {stock_option} from {start_date} to {end_date}")
        print(start_date)

    # Classifier Section
    st.sidebar.subheader("🧠 Choose a Technique: ")
    classifier_option = st.sidebar.radio(
        'Select a Classifier:',
        ['ID3', 'Naive Bayes', "GRADIENTBOOST","RandomForest"],
        index=0
    )
    if start_date != end_date:
        if classifier_option == 'ID3':
            st.header("🌳 ID3 Classifier Results")
            analysis = StockAnalysisid3(stock, start_date, end_date)
            accuracy = analysis.get_accuracy()
            try:
                st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
            except:
                None
            st.write("### 📊 Data Plot")
            analysis.plot_data()
            st.write("### 🌲 Decision Tree Visualization")
            try:
                analysis.plot_decision_tree()
            except:
                None

        elif classifier_option == 'Naive Bayes':
            st.header("🧠 Naive Bayes Classifier Results")
            nb_classifier = NaiveBayesClassifier(stock, start_date, end_date)
            accuracy = nb_classifier.get_accuracy()
            st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
            st.write("### Confusion Matrix")
            nb_classifier.display_confusion_matrix()
            st.write("### Classification Report")
            nb_classifier.display_classification_report()
        elif classifier_option == 'GRADIENTBOOST':
            st.header("📈 GradientBoost Analysis")
            stock_analysisgb =  StockAnalysisGb(stock, start_date, end_date)

            # Display model accuracy
            accuracy = stock_analysisgb.get_accuracy()
            if accuracy is not None:
                st.subheader(f"Model Accuracy: {accuracy:.2f}%")
            else:
                st.write("No accuracy available")

            # Plot stock data and indicators
            stock_analysisgb.plot_data()
        elif classifier_option == 'RandomForest':
            st.header("📈 RandomForest Analysis")
            stock_analysisrf = StockAnalysisrf(stock, start_date, end_date)

            # Display model accuracy
            accuracy = stock_analysisrf.get_accuracy()
            if accuracy is not None:
                st.subheader(f"Model Accuracy: {accuracy*100:.2f}%")
            else:
                st.write("No accuracy available")

            # Plot stock data and indicators
            stock_analysisrf.plot_data()


