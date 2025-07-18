import mlflow
import mlflow.sklearn
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os


# Hyperparameters
max_depth = 12

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("iris-dt")

with mlflow.start_run(run_name = "DTWITH max_depth = 12 and confusion matrix"):
    # Initialize the model using variables
    rf = DecisionTreeClassifier(max_depth=max_depth)

    # Train the model
    rf.fit(X_train, y_train)    

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric("accuracy_score",accuracy)
    mlflow.log_param("max_depth",max_depth)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Log artifact      
    mlflow.log_artifact("confusion_matrix.png") 
    
    mlflow.log_feedback(__file__)
    mlflow.sklearn.log_model(rf, "decision_tree")
    
    mlflow.set_tag("sahil","author")

 
    print("accuracy",accuracy)