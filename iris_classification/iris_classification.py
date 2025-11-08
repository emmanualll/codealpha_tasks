import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    """
    Main function to run the complete data science workflow.
    """
    

    print("Loading Iris dataset...")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    df = pd.DataFrame(data=X, columns=feature_names)
    df['species'] = y
    
    # Map target IDs to species names for plotting
    df['species_name'] = df['species'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})
    
    print("Data loaded successfully.")
    print(df.head())
    print("-" * 50)
    

    print("Generating Exploratory Data Analysis (EDA) plot...")
    # Create a pair plot to visualize relationships between features
    # 'hue' colors the dots based on the 'species_name'
    sns.pairplot(df, hue='species_name', diag_kind='kde')
    

    eda_plot_file = "iris_pair_plot.png"
    plt.savefig(eda_plot_file)
    print(f"EDA pair plot saved as '{eda_plot_file}'")
    plt.close() # Close the plot to free up memory
    print("-" * 50)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Split data into {len(X_train)} training and {len(X_test)} test samples.")
    
    # Scale the features (X data)
    # This standardizes features to have 0 mean and unit variance.
    # It's important for models like SVM and Logistic Regression.
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled successfully.")
    print("-" * 50)


    print("Training and evaluating models...")
    

    models = {
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Support Vector Machine (SVM)": SVC(kernel='linear'),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0.0
    

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)
        

        y_pred = model.predict(X_test_scaled)
        

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("-" * 50)
    print("Model Comparison (Accuracy):")
    # Sort results for a clean summary
    for name, accuracy in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {accuracy * 100:.2f}%")
    
    print(f"\nBest performing model: {best_model.__class__.__name__}")
    print("-" * 50)


    print(f"Generating detailed report for the best model ({best_model.__class__.__name__})...")
    

    y_pred_best = best_model.predict(X_test_scaled)
    

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred_best, target_names=target_names)
    print(report)
    

    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {best_model.__class__.__name__}')
    

    cm_plot_file = "iris_confusion_matrix.png"
    plt.savefig(cm_plot_file)
    print(f"Confusion Matrix plot saved as '{cm_plot_file}'")
    plt.close()
    print("-" * 50)
    print("Workflow complete. You now have 'iris_pair_plot.png' and 'iris_confusion_matrix.png' for your presentation.")


if __name__ == "__main__":
    main()