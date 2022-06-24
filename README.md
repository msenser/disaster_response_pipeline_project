# Disaster Response Pipeline Project

## Table of Contents

1. [Project Summary](#summary)
2. [Installation](#installation)
3. [Instructions](#instructions)
4. [File Descriptions](#files)

## Project Summary<a name="summary"></a>

This project analyzes disaster data and uses it to create a model for an API that classifies disaster messages.  
The project includes a web app that will allow an emergency responder to enter a new message and receive the classification results in different categories so they can send the messages to an appropriate disaster relief organization. The web app will also display visualizations of the data.   
It is also possible to read in own datasets to calculate a new model.

## Installation <a name="installation"></a>

- Python v3.* needed

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions <a name="files"></a>

- app
    - template
        - master.html (main page of web app)
        - go.html  (classification result page of web app)
    - run.py  (Flask file that runs app)

- data
    - disaster_categories.csv (data to process)
    - disaster_messages.csv  (data to process)
    - process_data.py
    - InsertDatabaseName.db (database to save clean data to)

- models
    - train_classifier.py
    - classifier.pkl  (saved model) 

