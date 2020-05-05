# ds-nd-disaster-response

## **Summary**

This project is part of the Udacity Data Scientist Nanodegree. 

The project consists of three parts:

### **1. ETL Pipeline**

The ETL pipeline uses two datasets as inputs which contain messages and already defined categories to those messages. <br> The pipeline merges both tables, cleans the data and finally loads the data into a SQLlite database.

### **2. ML Pipeline**

The ML pipeline uses CountVectorizer and TfidfTransfomer for the text data and predicts catgories with the help of an MultiOutputClassifier in connection with Random Forests. <br> GridSearch is used to find the best model for certain predefined parameters. The model is saved as a pickle file for further usage within the web app. 

### **3. Web App**

The web app allows for new text messages which are classified in the given categories using the ML model build in step 2.

## How to use

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database <br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves <br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`