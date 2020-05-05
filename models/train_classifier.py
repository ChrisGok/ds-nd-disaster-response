# import packages
import sys

import re
import pickle
import nltk
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load data 
def load_data(database_filepath):
    """
    Function loads SQL table from specified database_filepath
    
    Parameters:
        database_filepath (str): string to database.

    Returns:
        X values, y values, categories 
    """
    # connect to database and load SQL table
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_categories', con=engine)
    
    # prepare X and y data sets
    X = df['message'].values
    y = df.drop(['id', 'message', 'original','genre'],axis=1)
    y = y.astype(int)
    
    # get category names from y
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Function prepares text data using tokenization, lemmatization,
    removal of stop words, normalization 
    
    Parameters:
        text: string 

    Returns:
        clean_tokens: list of clean tokenized words
    """
    
    # initialize lemmatizer and define stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            # normalize case and remove punctuations
            clean_tok = re.sub(r"[^a-zA-Z0-9]"," ", tok.lower().strip())
            # lemmatize and remove stop words
            clean_tok = lemmatizer.lemmatize(clean_tok).lower().strip()
            clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Function set up NLP-Pipeline using CountVectorizer, TfidfTransformer and 
    MultiOutputClassifier(RandomForestClassifier) for the pipeline which is input for
    GridSearch.
    Only certain parameters are considered for GridSearch. 
    
    Parameters:
        None 

    Returns:
        model: model optimized with GridSearch
    """

    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50,100],
        'clf__estimator__min_samples_split': [3, 4],
        'text_pipeline__tfidf__use_idf': (True, False)
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Function to evaluate the model performance using sklearns classification_report

    Parameters:
        model: ML model
        X_test: test data set for dependent variable
        y_test: test data set for target variable
        category_names: list of names used for classification_report

    Returns:
        prints the results of the classification_report for all categories
    """
    
    y_pred = model.predict(X_test)
    
    for col in range(y_pred.shape[1]): 
        print('Result of {}:'.format(category_names[col]), '\n', classification_report(y_pred[:,col], y_test.iloc[:,col]))


def save_model(model, model_filepath):
    """
    Function saves the model to a pickle file.
    
    Parameters:
        model: ML model
        model_filepath: name / path of the saved model

    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()