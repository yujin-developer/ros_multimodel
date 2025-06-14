import rclpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from rclpy.node import Node
from std_msgs.msg import Float64, String


class IrisPublisher(Node):
    def __init__(self):
        super().__init__('iris_publisher')

        # Load dataset
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(current_dir, "Iris.csv")
        df = pd.read_csv(dataset_path,sep=",")
        
        # df
        df.isnull().sum()  # check missing value
        # df.columns  # 'Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'
        # df["Species"].value_counts()  # Iris-setosa, Iris-versicolor, Iris-virginica (Check Species values)

        # Remove Id colums (just numbers which means not helpful for training model)
        df.drop("Id", axis=1, inplace=True)

        # Encode categorical column to numerical labels
        species_mapping = { "Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2 }
        df["label"] = df["Species"].map(species_mapping)

        # Define X(features) and y(target)
        X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
        y = df["label"]

        # Visualize two parameters
        fig1 = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", palette="Set2")
        plt.title("Sepal Length vs Sepal Width")
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width")
        plt.grid(True)
        fig1.tight_layout()
        output_filename = "Iris_Result_1.pdf"
        output_path = os.path.abspath(output_filename)
        fig1.savefig(output_path, format='pdf')  # Save to PDF
        print(f"PDF saved successfully at: {output_path}")
        # plt.show()
        plt.close(fig1)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split dataset into training(70%), testing(30%)
        X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.3, random_state=42, stratify=y )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print("Total Dataset Size: ", len(df))
        print("Train Dataset Size: ", len(X_train))
        print("Test Dataset Size: ", len(X_test))

        # Make Publisher
        self.publisher_ = self.create_publisher(String, 'model_metrics', 10)
        self.timer = self.create_timer(5.0, self.prediction)
        self.index = 0

        self.predicted = False

    def prediction(self):
        if self.predicted:
            return
        
        # Prepare models
        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        # prepare dict to store metrics
        performance = {
            "Model": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": []
        }

        # Train, Predict, Evalutae Models
        msg = String()

        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            performance["Model"].append(name)
            performance["Accuracy"].append(accuracy_score(self.y_test, y_pred))
            performance["Precision"].append(precision_score(self.y_test, y_pred, average='macro'))
            performance["Recall"].append(recall_score(self.y_test, y_pred, average='macro'))
            performance["F1 Score"].append(f1_score(self.y_test, y_pred, average='macro'))
            
            # print("-----------------------------------------------------------")
            # print(f"{name} Classification Report:")
            # print(classification_report(self.y_test, y_pred))

            msg.data += "-----------------------------------------------------------\n"
            msg.data += f"{name} Classification Report:\n"
            msg.data += classification_report(self.y_test, y_pred) + "\n"

        # dict to DataFrame and round values
        result_df = pd.DataFrame(performance).round(2)

        # print model performance comparison
        # print("-----------------------------------------------------------")
        # print("Model Performance Comparison:")
        # print(result_df.to_string(index=False))
        
        # append model performance comparison
        msg.data+= "-----------------------------------------------------------\n"
        msg.data += "Model Performance Comparison:\n"
        msg.data += result_df.to_string(index=False)

        # Visualization bar charts for comparison model metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        fig2, axes = plt.subplots(2, 2, figsize=(10, 6))  # 2X2 = 4 subplots
        axes = axes.flatten()  # to be 1D
        for i, metric in enumerate(metrics):
            axes[i].bar(result_df["Model"], result_df[metric], color='lightblue')
            axes[i].set_title(metric)
            axes[i].set_ylim(0.85, 1.0)
            for j, value in enumerate(result_df[metric]):
                axes[i].text(j, value, f"{value:.2f}", ha='center', va='bottom')

        plt.suptitle("Model Performance Comparison", fontsize=16)
        fig2.tight_layout()
        output_filename = "Iris_Result_2.pdf"
        output_path = os.path.abspath(output_filename)
        fig2.savefig(output_path, format='pdf')  # Save to PDF
        print(f"PDF saved successfully at: {output_path}")
        # plt.show()
        plt.close(fig2)

        # Make Publisher
        self.publisher_.publish(msg)
        self.get_logger().info("Published Message(Classification Report).")

        self.predicted = True

def main(args=None):
    rclpy.init(args=args)
    node = IrisPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

