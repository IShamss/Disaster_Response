import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    '''
    This function loads the data from the SQL database and prepares it for machine learning modeling.

    Args:

    database_filepath (str): file path of the database containing the cleaned data
    Returns:

    X (numpy array): an array containing the messages
    Y (numpy array): an array containing the labels
    category_names (numpy array): an array containing the names of the categories in Y
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages',engine)
    X = df['message'].values
    Y = df.iloc[:,5:].values
    category_names = df.iloc[:,5:].columns 
    return (X,Y, category_names)
    


def tokenize(text):
    '''
    Tokenize the input text by removing non-alphanumeric characters, converting to lowercase, and lemmatizing each word.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of cleaned and lemmatized tokens extracted from the input text.

    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    # Lemmatize the stemming
    lemmatizer = WordNetLemmatizer()
#     lemmed = [lemmatizer.lemmatize(w).lower().strip() for w in text if w not in stopwords]
#     stemmed = [PorterStemmer().stem(w).strip() for w in lemmed]
    result = []
    for token in tokens:
        result.append(lemmatizer.lemmatize(token).lower().strip())
    return result


def build_model():
    '''
    Returns a pipeline that processes text data and trains a multi-output classification model using Random Forest Classifier.
    
    Returns:
    pipeline (Pipeline): A pipeline that processes text data and trains a multi-output classification model.
    '''
    pipeline = Pipeline([
        ('text_pipeline',Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer())
        ])),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the performance of a given model on a test set of input features X_test and true labels Y_test. It prints a classification report and accuracy score for each category in category_names.

    Arguments:

    model: the trained machine learning model to be evaluated
    X_test: the test set of input features (array-like or sparse matrix)
    Y_test: the true labels for the test set (array-like)
    category_names: a list of category names to be used in the classification report
    Returns:

    None

    '''
    y_pred = model.predict(X_test)
    
    y_test_df = pd.DataFrame(Y_test,columns=category_names)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)

    print(classification_report(y_test_df,y_pred_df,target_names=category_names))
    
    labels = np.unique(y_pred)
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    '''
    The save_model function saves the trained model as a pickle file to a specified file path.

    Parameters:

    model: the trained model to be saved
    model_filepath: the file path to save the model to
    Returns:

    None
    '''
    with open(model_filepath,"wb") as f:
        pickle.dump(model,f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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