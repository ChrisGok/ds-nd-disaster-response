# ds-nd-disaster-response

## **Summary**

This project is part of the Udacity Data Scientist Nanodegree. 

The project consists of three parts:

### **1. ETL Pipeline**

The ETL pipeline uses two datasets as inputs which contain messages and already defined categories to those messages. <br> The pipeline merges both tables, cleans the data and finally loads the data into a SQLlite database.

### **2. ML Pipeline**

The ML pipeline uses CountVectorizer and TfidfTransfomer for the text data and predicts catgories with the help of an MultiOutputClassifier in connection with Random Forests. <br> GridSearch is used to find the best model for certain predefined parameters. GridSearch takes a while to run, so remember to adjust the parameters. The model is saved as a pickle file for further usage within the web app. <br>
Since the data is unbalanced 

### **3. Web App**

The web app allows for new text messages which are classified in the given categories using the ML model build in step 2. <br>
A brief overview is provided for the underlying data as shown in the figures of the app. Since we deal with an unbalanced data set having many observations for certain categories while only a few data points for other categories. Hence, the results of the classification test should be interpreted with caution.

## How to use

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database <br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves <br>
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app. <br>
    `python run.py`
    you can use `localhost:3001`

## License
The data was provided by Udactiy Partner FigureEight. The web app code was only slightly adjusted and was also provided by Udacity.