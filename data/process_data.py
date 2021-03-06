import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sympy import O

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data from csv.-files and merge it to one data frame

    Input:
    messages_filepath - filepath to messages csv file
    categories_filepath - filepath to categories csv file

    Returns:
    merged dataframe containing messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Clean the data from dataframe and return a dataframe with a single column for each category with only binary values

    Input:
    df - merged dataframe containing messages and categories

    Returns:
    cleaned dataframe with a single column for each category with only binary values
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column]) 
        # drop all rows with unexpected values 
        categories.drop(categories[(categories[column] != 0) & (categories[column] != 1)].index, inplace=True)
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join="inner")
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save the data from dataframe to an SQL database

    Input:
    df - database to be saved
    database_filename - filename of the SQL database
    '''
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    '''
    Read the dataset, clean the data, and then store it in a SQLite database
    '''
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