# Disaster Response Pipeline Project

## Motivation
Natural disasters like earthquakes, hurricanes, and floods can be devastating, leading to loss of life, property damage, and displacement of people. During such emergencies, it's crucial to have an effective communication system that can quickly identify the needs of the affected people and coordinate the response efforts. However, with the increasing volume of messages sent through various channels like social media, emails, and text messages, it's challenging for disaster response organizations to manually sift through them to identify the ones that require immediate attention.

To address this challenge, machine learning pipelines can be trained to classify messages based on their content, enabling disaster response organizations to prioritize their response efforts and save lives.


## File Description

 * [tree-md](./tree-md)
 * [app](./app)
   * [templates](./app/templates)
     * [go.html](./app/templates/go.html) # main page of web app
     * [master.html](./app/templates/master.html) # classification result page of web app
   * [run.py](./app/run.py) # Flask file that runs app
 * [data](./data)
   * [disaster_categories.csv](./data/disaster_categories.csv) # data to process
   * [disaster_messages.csv](./data/disaster_messages.csv) # data to process
   * [DisasterResponse.db](./data/DisasterResponse.db) # database to save clean data to
   * [process_data.py](./data/process_data.py) # script for data preprocessing
 * [models](./models)
   * [train_classifier.py](./models/train_classifier.py) # script for training ML model
   * [classifier.pkl](./models/classifier.pkl)  # saved model
 * [README.md](./README.md)
 
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

