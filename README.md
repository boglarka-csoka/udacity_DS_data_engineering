# Repository for the udacity data science course, project 2

## Overview
This project is part of the Udacity Data Science Nanodegree.
In this project, we build and optimize an NLP pipeline using the Python. Our goal is to use our model to classify messages with the help of random forest. 
In the screen the user can add a text message and the program will categorize it.

## This repository contains the followings:
- app
  - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - DisasterResponse.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 

## Summary
This dataset contains help messages text data.To preprocess it, we first create 'DisasterResponse.db': cleaning and mergeing 2 datasets (disaster_categories.csv,disaster_messages.csv). Then create a model ('classifier.pkl') to predict the category of new (unseen) text messages.
This model is trained and optimized with random forest estimator. I evaluate the model using precision, recall and f1 score for each label.
The user can type a text message and the program will categorize it.
