import rclpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from rclpy.node import Node
from std_msgs.msg import Float64, String


class FruitPublisher(Node):
    def __init__(self):
        super().__init__('fruit_publisher')

        # Load dataset
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(current_dir, "fruits_weight_sphercity.csv")
        df = pd.read_csv(dataset_path,sep=",")
        df.isnull().sum()  # check missing value

        # One-hot encoding for categorical column
        df_encoded = pd.get_dummies(df, columns=["Color"], drop_first=True)
        # df_encoded
        # df_encoded.columns  # check new dummy columns

        # Define features(X)
        X = df_encoded[["Weight", "Sphericity", "Color_Greenish yellow", "Color_Orange", "Color_Red", "Color_Reddish yellow"]]

        # Define labels(y) which is target feature
        labels_mapping = {"apple": 0, "orange": 1}
        y = df["labels"].map(labels_mapping)

        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Visualize relationship between two features(Weight, Sphericity)
        fig1 = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="Weight", y="Sphericity", hue="labels")
        plt.title("Weight vs Sphericity")
        plt.xlabel("Weight")
        plt.ylabel("Sphericity")
        plt.grid(True)
        output_filename = "Fruit_Result_1.pdf"
        output_path = os.path.abspath(output_filename)
        fig1.tight_layout()
        fig1.savefig(output_path, format='pdf')  # Save visualization file as pdf
        print(f"PDF saved successfully at: {output_path}")
        # plt.show()
        plt.close(fig1)


        # Split dataset into training(70%), testing(30%)
        X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.3, random_state=42, stratify=y )
        self.y_test = y_test
        self.X_test = X_test
        
        print("Total Dataset Size: ", len(df))
        print("Train Dataset Size: ", len(X_train))
        print("Test Dataset Size: ", len(X_test))

        # Train model linear classifier (Logistic Regression)
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        # Make Publisher
        self.publisher_ = self.create_publisher(String, 'model_metrics', 10)
        self.timer = self.create_timer(5.0, self.prediction)
        self.index = 0

        self.predicted = False

    def prediction(self):
        if self.predicted:
            return
        
        # Predict model
        y_pred = self.model.predict(self.X_test)

        # Evaluate model performance
        report = classification_report(self.y_test, y_pred)
        msg = String()
        msg.data = ( f"Classification Report:\n{report}" )

        # Confusion Matrix 
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        fig2 = plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["apple", "orange"], yticklabels=["apple", "orange"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        output_filename = "Fruit_Result_2.pdf"
        output_path = os.path.abspath(output_filename)
        fig2.tight_layout()
        fig2.savefig(output_path, format='pdf')  # Save visualization file as pdf
        print(f"PDF saved successfully at: {output_path}")
        # plt.show()
        plt.close(fig2)
        
        # Make Publisher
        self.publisher_.publish(msg)
        self.get_logger().info("Published Message(Classification Report).")

        self.predicted = True

def main(args=None):
    rclpy.init(args=args)
    node = FruitPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()