import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='campusx-official', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/campusx-official/mlflow-dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 1
n_estimators = 100

# apply mlflow

mlflow.set_experiment('iris-rf')

with mlflow.start_run():

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")

    # mlflow code
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(rf, "random forest")

    mlflow.set_tag('author','rahul')
    mlflow.set_tag('model','random forest')

    print('accuracy', accuracy)

