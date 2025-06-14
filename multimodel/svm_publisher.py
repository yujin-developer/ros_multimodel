import rclpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from mpl_toolkits.mplot3d import Axes3D
from rclpy.node import Node
from std_msgs.msg import Float64, String


class SvmPublisher(Node):
    def __init__(self):
        super().__init__('svm_publisher')
        
        # Load dataset(Breast Cancer)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(current_dir, "Breast_Cancer.csv")
        df_breast_cancer = pd.read_csv(dataset_path,sep=",")
        
        # df_breast_cancer
        df_breast_cancer.isnull().sum()  # check missing value
        # df_breast_cancer.columns  # check columns names 
        # df_breast_cancer["diagnosis"].value_counts()  # check diagnosis values

        # Remove id colums (not helpful to train model)
        df_breast_cancer.drop("id", axis=1, inplace=True)

        # Visualization with 2 features
        features = ["radius_mean", "texture_mean"]
        fig1 = plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_breast_cancer,
            x=features[0],
            y=features[1],
            hue="diagnosis",
            palette={'B': "blue", 'M': "red"},
            alpha=0.7,
            edgecolor='k'
        )

        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title("Data Distribution (radius_mean and texture_mean)")
        plt.legend(title="Diagnosis")
        plt.grid(True)
        fig1.tight_layout()
        output_filename = "SVM_Result_1.pdf"
        output_path = os.path.abspath(output_filename)
        fig1.savefig(output_path, format='pdf')  # Save to PDF
        print(f"PDF saved successfully at: {output_path}")
        # plt.show()
        plt.close(fig1)

        # Encoding to numerical values (target label mapping B: 0, M: 1) 
        df_breast_cancer["diagnosis"] = df_breast_cancer["diagnosis"].map({"B": 0, "M": 1})

        # Define Features and target
        y_breast_cancer = df_breast_cancer["diagnosis"]
        X_breast_cancer = df_breast_cancer.drop("diagnosis", axis=1)


        # Normalize features
        X_breast_cancer = StandardScaler().fit_transform(X_breast_cancer)

        # Split dataset into training(70%), testing(30%)
        X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.3, random_state=42, stratify=y_breast_cancer)
        self.X_bc_train = X_bc_train
        self.X_bc_test = X_bc_test
        self.y_bc_train = y_bc_train
        self.y_bc_test = y_bc_test
        
        print("Total Breast Cancer Dataset Size: ", len(df_breast_cancer))
        print("Train Breast Cancer Dataset Size: ", len(X_bc_train))
        print("Test Breast Cancer Dataset Size: ", len(X_bc_test))

        # Define model(SVM Linear Classifier)
        self.svm_breast_cancer = SVC(kernel='linear')

        # Train model
        self.svm_breast_cancer.fit(X_bc_train, y_bc_train)


        ##############################################################################################################################
        # Load penguin dataset
        df_penguin = pd.read_csv("Penguin.csv")
        # penguin_df
        # penguin_df.isnull().sum()  # check missing value
        # penguin_df.columns  # check columns names ['species', 'island', 'culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']
        # penguin_df["species"].value_counts()  # check species values(Adelie, Gentoo, Chinstrap)
        # penguin_df.info()  # check categorical columns(species,island,sex)
        # penguin_df["island"].value_counts()
        # penguin_df["sex"].value_counts()

        # Drop missing values
        df_penguin.dropna(inplace=True)

        # Data 3D Visualization
        fig2 = plt.figure(figsize=(10, 7))
        axes3d = fig2.add_subplot(111, projection='3d')

        # Extract three features
        x = df_penguin["culmen_length_mm"]
        y = df_penguin["culmen_depth_mm"]
        z = df_penguin["flipper_length_mm"]
        species = df_penguin["species"]

        # Assign color map
        species_unique = species.unique()
        colors = dict(zip(species_unique, ['red', 'green', 'blue']))
        color_values = species.map(colors)

        # Make Plot
        axes3d.scatter(x, y, z, c=color_values, edgecolor='k', alpha=0.7)
        axes3d.set_xlabel('Culmen Length (mm)')
        axes3d.set_ylabel('Culmen Depth (mm)')
        axes3d.set_zlabel('Flipper Length (mm)')
        axes3d.set_title("Penguin Species Distribution (3D)")
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=sp, markerfacecolor=colors[sp]) for sp in species_unique], title='Species')
        plt.tight_layout()
        output_filename = "SVM_Result_2.pdf"
        output_path = os.path.abspath(output_filename)
        fig2.savefig(output_path, format='pdf')  # Save to PDF
        print(f"PDF saved successfully at: {output_path}")
        # plt.show()
        plt.close(fig2)

        # One-hot encoding for categorical columns(species, island, sex)
        df_penguin = pd.get_dummies(df_penguin, columns=["island", "sex"], drop_first=False)

        # Split features and labels
        y_penguin = df_penguin["species"]
        X_penguin = df_penguin.drop("species", axis=1)

        # Normalize the features
        X_penguin = StandardScaler().fit_transform(X_penguin)

        # Split into train and test sets
        X_pg_train, X_pg_test, y_pg_train, y_pg_test = train_test_split(X_penguin, y_penguin, test_size=0.3, random_state=42, stratify=y_penguin)
        self.X_pg_train = X_pg_train
        self.X_pg_test = X_pg_test
        self.y_pg_train = y_pg_train
        self.y_pg_test = y_pg_test
        
        print("Total Penguin Dataset Size: ", len(df_penguin))
        print("Train Penguin Dataset Size: ", len(X_pg_train))
        print("Test Penguin Dataset Size: ", len(X_pg_test))

        # Train SVM with linear kernel
        self.svm_penguin = SVC(kernel='linear')
        self.svm_penguin.fit(X_pg_train, y_pg_train)


        # Make Publisher
        self.publisher_ = self.create_publisher(String, 'model_metrics', 10)
        self.timer = self.create_timer(5.0, self.prediction)
        self.index = 0

        self.predicted = False

    def prediction(self):
        if self.predicted:
            return    
        
        msg = String()

        # Breat Cancer Predict and evaluate
        y_bc_pred = self.svm_breast_cancer.predict(self.X_bc_test)
        accuracy_bc = accuracy_score(self.y_bc_test, y_bc_pred)
        precision_bc = precision_score(self.y_bc_test, y_bc_pred)
        recall_bc = recall_score(self.y_bc_test, y_bc_pred)
        f1_bc = f1_score(self.y_bc_test, y_bc_pred)

        # print("-----------------------------------------------------------")
        # print("Breast Cancer Classification Report:")
        # print(classification_report(self.y_bc_test, y_bc_pred, target_names=['B', 'M']))

        # Breast Cancer Classification Report
        msg.data += "-----------------------------------------------------------\n"
        msg.data += "Breast Cancer Classification Report:\n"
        msg.data += classification_report(self.y_bc_test, y_bc_pred, target_names=['B', 'M']) + "\n"
       
        # Penguin Predict and evaluate
        y_pg_pred = self.svm_penguin.predict(self.X_pg_test)
        accuracy_pg = accuracy_score(self.y_pg_test, y_pg_pred)
        precision_pg = precision_score(self.y_pg_test, y_pg_pred, average='macro')
        recall_pg = recall_score(self.y_pg_test, y_pg_pred, average='macro')
        f1_pg = f1_score(self.y_pg_test, y_pg_pred, average='macro')

        # print("-----------------------------------------------------------")
        # print("Penguin Classification Report:")
        # print(classification_report(self.y_pg_test, y_pg_pred))

        # Penguin Classification Report
        msg.data += "-----------------------------------------------------------\n"
        msg.data += "Penguin Classification Report:\n"
        msg.data += classification_report(self.y_pg_test, y_pg_pred) + "\n"

        # Comparison Metrics with two datasets
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        # each model/dataset performance
        breast_cancer_performance = [accuracy_bc, precision_bc, recall_bc, f1_bc]
        penguin_performance = [accuracy_pg, precision_pg, recall_pg, f1_pg]

        # Plot
        fig3 = plt.figure(figsize=(8, 5))
        plt.plot(metrics, breast_cancer_performance, marker='o', label='Breast Cancer')
        plt.plot(metrics, penguin_performance, marker='s', label='Penguin')

        plt.title('Performance Comparison of SVM Model')
        plt.ylabel('Score')
        plt.ylim(0.7, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_filename = "SVM_Result_3.pdf"
        output_path = os.path.abspath(output_filename)
        fig3.savefig(output_path, format='pdf')  # Save to PDF
        print(f"PDF saved successfully at: {output_path}")
        # plt.show()
        plt.close(fig3)

        # Make Publisher
        self.publisher_.publish(msg)
        self.get_logger().info("Published Message(Classification Report).")

        self.predicted = True

def main(args=None):
    rclpy.init(args=args)
    node = SvmPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
