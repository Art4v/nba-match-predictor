# import necessary libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split



''' Parse Synthetic Data CSV '''

csv_path = os.path.join(os.path.dirname(__file__), 'data', 'games.csv') # get path to games.csv in data directory
df = pd.read_csv(csv_path) # read the csv file into a pandas dataframe


''' Training Random Forest Classifier '''

# split data into features and target
X = df.drop(columns=['WL'])
y = df['WL']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# build random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
 
# train the model
rf.fit(X_train, y_train)


''' Model Testing and Evaluation '''

# make predictions on the test set
y_pred = rf.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# print evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")