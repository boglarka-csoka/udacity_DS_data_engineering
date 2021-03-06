import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads 2 datasets and merges them.

    This function loads messages and categories from csv files and merges them to one dataframe.

    Parameters
    ----------
    messages_filepath : str
        Path to disaster_messages.csv.
    categories_filepath : str
        Path to disaster_categories.csv.

    Returns
    -------
    dataframe
        Merged.
    """
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath, sep=',')
    #print(categories.head())
    df= messages.merge(right=categories,on='id')
    return df


def clean_data(df):
    """
    Preprocesses the dataframe.

    This function does some needed changes in the dataframe for further usage. 
    The steps: 
    - Create a new 'categories' table with proper column names and values as 0/1.
    - Drop the original 'categories' column from the dataframe and concatenate the dataframe with the newly created 'categories' table
    - Drop the duplicates

    Parameters
    ----------
    df : dataframe
        The dataframe that we would like to preprocess.

    Returns
    -------
    dataframe
        The preprocessed dataframe.
    """

    categories = pd.DataFrame(df.categories.str.split(pat=';',expand=True))
    row = categories.iloc[0]
    y = lambda value : value[0:len(value)-2]

    new_colnames=[]
    for i in range(len(row)):
        colname=y(row[i])
        new_colnames.append(colname)
        
    category_colnames = new_colnames
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str)
        # set each value to be the last character of the string
        categories[column]=[x[-1:] for x in categories[column]]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)
    # concatenate the 2 dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df=df.drop_duplicates() 
    # drop related column's 2 values
    df = df[df.related != 2]

    return df





def save_data(df, database_filename):
    """
    Saves the dataframe.

    This function saves the dataframe to sqlite.

    Parameters
    ----------
    df : dataframe
        The dataframe that we would like to save.
    database_filename : str
        The dataframe name that we would like to use for saving.

    Returns
    -------
    None
    """

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(database_filename, engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
