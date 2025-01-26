import streamlit as st
from iD3_classifier import StockAnalysis
from Naive_Bayes_classifier import NaiveBayesClassifier
from linear_Regression import StockPredictor
from linear_reg import StockPricePredictor
# Set the title for the application
st.title("Stock Analysis Dashboard")

# Sidebar with a dropdown menu for selecting stocks
st.sidebar.title("Select Your Stock")

stock_option = st.sidebar.selectbox(
    'Choose a Stock:',
    ['Stocks','NIFTY', 'VBL', 'CoalIndia', 'ITC']  # Adding an empty string as the first option
)

# Display the selected stock option in the main section
if stock_option:
    st.header(f"You selected: {stock_option}")

    # Map the selected stock to its symbol
    if stock_option == 'NIFTY':
        stock = "^NSEI"
    elif stock_option == 'VBL':
        stock = "VBL.NS"
    elif stock_option == 'CoalIndia':
        stock = "COALINDIA.NS"
    elif stock_option == 'ITC':
        stock = "ITC.NS"
    else:
        None

    start_date = st.sidebar.date_input("Enter the start Date: ")
    end_date = st.sidebar.date_input("Enter the End Date: ")
    st.write(start_date, "  ", end_date)
    # Sidebar with a dropdown menu for selecting classifiers
    st.sidebar.title("Select Your Classifier")
    classifier_option = st.sidebar.selectbox(
        'Choose a Classifier:',
        ['Classifiers', 'ID3', 'Naive Bayes']  # Adding an empty string as the first option
    )

    # Add logic for the selected classifier
    if classifier_option == 'ID3':
        st.header("You selected ID3 Classifier")
        analysis = StockAnalysis(stock)  # Everything gets initialized and processed
        accuracy = analysis.get_accuracy()  # Fetch accuracy
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        analysis.plot_data()  # Display plots
        analysis.plot_decision_tree()

    elif classifier_option == 'Naive Bayes':
        st.write("You selected Naive Bayes Classifier")
        nb_classifier = NaiveBayesClassifier(stock)  # Initialize and run classification

        # Get and display accuracy
        accuracy = nb_classifier.get_accuracy()
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

        # Display confusion matrix
        nb_classifier.display_confusion_matrix()

        # Display classification report
        nb_classifier.display_classification_report()

    else:
        st.write("Please select a classifier to continue.")

# Sidebar for regression option
    st.sidebar.title("Select Regression Option")
    regression_option = st.sidebar.selectbox(
        'Choose:',
        ['Regressor', 'Linear Regression']  # Adding an empty string as the first option
    )

    # Display the selected regression option in the main section
    if regression_option == 'Linear Regression' and stock_option:
        predictor = StockPricePredictor(stock, start_date, end_date)
        predictor.predicting()
        predictor.plot_actual_vs_predicted()
        predictor.plot_residuals()
        predictor.evaluate_model()
        predictor.model_summary()


