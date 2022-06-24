#import libraries and load data from database
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline #, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Load the data from SQL Database and transform it to a data frame. Divide dataframe into X and Y.

    Input:
    database_filepath - filepath to the SQL database

    Returns:
    X - the messages to be categorized
    Y - the categories of the messages
    col_names - the names of the categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', con= engine)
    X = df['message'].values
    Y = df.drop(columns=['id','message','original','genre'])
    col_names=list(Y.columns)
    return X,Y,col_names


def tokenize(text):
    '''
    Tokenize text to process text data

    Input:
    text - a string to tokenize

    Returns:
    the tokens of the text
    '''
    tokens = []
    for tok in word_tokenize(text):
        tokens.append(WordNetLemmatizer().lemmatize(tok.lower().strip()))
    return tokens


def build_model():
    '''
    Build the model using a pipleine and GridSearch.

    Returns:
    model with defined GridSearch parameters
    '''
    # Build a machine learning pipeline
    pipeline = Pipeline([           
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('mclf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    #Improve model with Grid Search
    parameters = {
        'tfidf__smooth_idf': (True, False),
        'mclf__estimator__n_estimators': (25, 50)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model and print accuracy, precision, recall and f1-score for each category

    Input:
    model - transformed and fitted model
    X_test - the X values to be tested (the messages)
    Y_test - the Y values with the correct categories
    category_names - the names of the categories as list
    '''
    # Evaluate and print results
    Y_pred = pd.DataFrame(model.predict(X_test))
    for i in range(35):
        print(category_names[i])
        cl = classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i],output_dict=True)
        try:
            print("      Accuracy: {:.4f}%    Precision:{:.4f}%    Recall: {:.4f}%    f1: {:.4f}".
                format(cl.get('accuracy'),cl.get('1').get('precision'),
                        cl.get('1').get('recall'),cl.get('1').get('f1-score')))
        except:
            print("      Accuracy: {:.4f}%    Precision:{}    Recall: {}    f1: {}".
                format(cl.get('accuracy'),'   -   ','   -   ','   -   '))


def save_model(model, model_filepath):
    '''
    Saves the model to a pickle-file to a defined path.

    Input:
    model - transformed and fitted model
    model_filepath - the path of the pickle file
    '''
    # Save model in Pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    split the data into a training set and a test set. 
    create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories
    export your model to a pickle file
    '''
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
              'train_classifier.py data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()