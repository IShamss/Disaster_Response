import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads two CSV files, one containing messages and the other containing         categories, and merges them on the "id" column. It returns a pandas DataFrame               containing the combined data.

    Args:
    messages_filepath (str): The file path of the messages CSV file.
    categories_filepath (str): The file path of the categories CSV file.

    Returns:
    pandas DataFrame: A DataFrame containing the merged data from the two CSV files.
    '''
    messages = pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)
    df = pd.merge(messages,categories,on='id')
    return df



def clean_data(df):
    """
    Clean the dataframe to get it ready for analysis. 
    This function extracts categories from the 'categories' column 
    and converts them into separate columns with binary values. 
    It also drops duplicates.
    
    Args:
    df (pandas.DataFrame): The input DataFrame containing messages and categories
    
    Returns:
    pandas.DataFrame: A cleaned version of the input DataFrame with binary category columns     and no duplicates
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = list(set([x[:-2] for x in row.values]))
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    return df.drop_duplicates()


def save_data(df, database_filepath):
    '''
    Args:

    df (Pandas DataFrame): The DataFrame to be saved into the SQLite database.
    database_filepath (str): The filepath of the database where the DataFrame will be           saved.
    Returns:

    None. The function only saves the DataFrame into the SQLite database.
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('Messages', engine, index=False,if_exists='replace')  


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