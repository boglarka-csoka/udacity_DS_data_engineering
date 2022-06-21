from email import message
from pickletools import read_decimalnl_long
import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])
import re
#from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score,precision_score
import pickle



def load_data(database_filepath):
    
    """
    Loads the DisasterResponse dataset and selects the X and Y parts of the dataset.

    Parameters
    ----------
    database_filepath : str
        Path to DisasterResponse.db.

    Returns
    -------
    dataframe
    X variable
    Y variables
    category names
    """

    engine = create_engine('sqlite:///'+ database_filepath)
    df=pd.read_sql_table(database_filepath, engine)
    #print(df.head())
    X= df['message']
    Y= df.loc[:, df.columns[4:]]
    category_names=df.columns[4:]
    #print(category_names)
    #print("Length of dataframe: ",len(df))
    return X,Y, category_names

def tokenize(text):

    """
    Tokenize the given text.

    This function clears and tokenize the given text.

    Parameters
    ----------
    text : str
        The text which need to be tokenized.

    Returns
    -------
    clean tokens
        Cleaned words.
    """

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation characters
    #text = text.replace("[", ".")
    #text = text.replace("]", ".")
    #text = text.replace("{", ".")
    #text = text.replace("}", ".")
    #text = text.replace("(", ".")
    #text = text.replace(")", ".")
    #print(text)
    text = re.sub(r'[\W_]+', " ", text)  #r"[^a-zA-Z0-9"
    words_in_text=word_tokenize(text)

    clean_tokens = []
    for tok in words_in_text:
        clean_tok = WordNetLemmatizer().lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():

    """
    Build the model.

    This function creates the pipeline and optimizes a hyperparameter. I only left 1 hyperparameter because it requires a lot of time.

    Parameters
    ----------

    Returns
    -------
    pipeline
         Gives back a pipeline with optimized parameters.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Only left 1 parameter because of the running time
    parameters = {
        'clf__estimator__n_estimators': [70, 140], # number of trees in the forest
        #'clf__estimator__min_samples_split': [2, 3, 4],
        #'clf__estimator__class_weight': ["balanced", "balanced_subsample","None"]
    }

    pipeline_cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline_cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
    Evaluates the model.

    This function gets the model and tries it on a test set.

    Parameters
    ----------
    model : model
        The model given by the built_model() function.
    X_test : dataframe
        The messages variable from the loaded data.
    Y_test : dataframe
        The valid classifications for the data.
    category_names : list
        Names of possible categories.

    Returns
    -------
    Nothing, but prints out the F1 score, Precision and Recall metrics.
    """

    # predict on test data
    Y_pred = model.predict(X_test)

    # Evaluate based on precision, recall and f1 score
    for cat in category_names:
        f1score= f1_score(Y_test[cat],Y_pred[cat])
        recall=recall_score(Y_test[cat],Y_pred[cat])
        precision=precision_score(Y_test[cat],Y_pred[cat])
        print('For category ', cat, '\n F1 score: ',f1score,' Recall: ',recall,' Precision: ',precision)


def save_model(model, model_filepath):

    """
    Saves the model.

    This function saves the model to the given path.

    Parameters
    ----------
    model : model
        The model that we would like to save.
    model_filepath : str
        The path where we would like to save our model.

    Returns
    -------
    None
    """

    pickle.dump(model, open(model_filepath, 'wb'))


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

        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        

        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()