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

def load_data(database_filepath):
    """Description"""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_categories', con=engine)
    X = df['message'].values
    y = df.drop(['id', 'message', 'original','genre'],axis=1)
    y = y.astype(int)
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """Description"""
    # initialize lemmatizer and define stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    # normalize case and remove punctuations
    # text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            # lemmatize and remove stop words
            clean_tok = re.sub(r"[^a-zA-Z0-9]"," ", tok.lower().strip())
            # clean_tok = [lemmatizer.lemmatize(word) for word in clean_tok if word not in stop_words]
            clean_tok = lemmatizer.lemmatize(clean_tok).lower().strip()
            clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """Description"""

    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10,20],
        'clf__estimator__min_samples_split': [2, 3]
        #'text_pipeline__vect__ngram_range': ((1, 1),(1,2)),
        #'text_pipeline__tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Description"""
    
    y_pred = model.predict(X_test)
    
    for col in range(y_pred.shape[1]): 
        print('Result of {}:'.format(category_names[col]), '\n', classification_report(y_pred[:,col], y_test.iloc[:,col]))


def save_model(model, model_filepath):
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