# import necessary libraries
import os
import pandas as pd

''' Parse Synthetic Data CSV '''
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'games.csv') # get path to games.csv in data directory
df = pd.read_csv(csv_path) # read the csv file into a pandas dataframe


''' Training Random Forest Classifier '''
# split data into features and target
X = df.drop(columns=['WL'])
y = df['WL']