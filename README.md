# Network Intrusion Detection

This project implements a Network Intrusion Detection System (NIDS) using machine learning techniques. The goal is to detect malicious network activities by classifying network traffic as either normal or indicative of an attack.

The NIDS project supports two classification tasks:
1. **Multi-Class Classification**: Identifying specific types of network behavior or attacks.
   - Classes include:
     - **Normal**: Normal Network Functionality
     - **TCP-SYN**: TCP-SYN Flood
     - **PortScan**: Port Scanning
     - **Overflow**: Flow Table Overflow
     - **Blackhole**: Blackhole Attack
     - **Diversion**: Traffic Diversion Attack

2. **Binary Classification**: Detecting whether network traffic is normal or indicative of an attack.
   - Classes include:
     - **Normal**: Normal Network Functionality
     - **Attack**: Network Intrusion

---

## Dataset

The project utilizes the University of Nevada - Reno Intrusion Detection Dataset (UNR-IDD), which is used for research in intrusion detection systems. The dataset is publicly available and provides detailed labeled instances of network traffic, which include various types of attacks and normal behavior.

### Dataset Overview
The dataset contains multiple network traffic instances that can be used to train machine learning models for both multi-class and binary classification tasks.

For more details about the dataset and to download it, visit the [official website](https://www.tapadhirdas.com/unr-idd-dataset).

---

## Data Preparation

### Load and Inspect the Dataset
The dataset is loaded from a CSV file into a Pandas DataFrame. Initial inspection is done to display the first few rows and analyze the distribution of different labels.

### Split the Dataset
The dataset is split into training, validation, and test sets using a stratified split to ensure proportional representation of the classes. 

- **Multi-Class Classification Split**:
   - Training set: 70%
   - Validation set: 15%
   - Test set: 15%

- **Binary Classification Split**:
   - Training set: 70%
   - Validation set: 15%
   - Test set: 15%

The rationale for separate splitting for binary and multi-class classification tasks includes:
1. **Preserving Label Distribution**: The `stratify` parameter ensures balanced representation.
2. **Tailored Handling of Imbalanced Data**: Binary and multi-class classification tasks have different levels of label imbalance.
3. **Independent Model Training and Evaluation**: Enables separate optimization for each classification task.
4. **Avoiding Data Leakage**: Ensures no data leakage between training and testing.

### Feature and Target Variable Preparation
Features and target variables are separated for training, validation, and testing. The target columns 'Label' and 'Binary Label' are dropped from the feature sets.

### Categorical Feature Encoding
`LabelEncoder` is used to convert categorical features into numerical format. Categorical columns such as 'Switch ID' and 'Port Number' are encoded.

### Dropping Columns with a Single Unique Value
Columns with a single unique value are identified and dropped from the datasets to eliminate redundant features.

### Scaling Numeric Features
`StandardScaler` is used to standardize the features to have zero mean and unit variance, which is essential for many machine learning algorithms.

---

## Exploratory Data Analysis

### Analyze Feature Distributions
The distribution of values in specific features is analyzed by counting occurrences and visualizing the first few rows to understand the data structure.

### Inspect Unique Values in Numeric Features
The unique values and the count of unique values for each feature are displayed, providing insights into the data distribution.

### Visualize Numeric Features with Boxplots
Boxplots are generated for each numeric feature to visualize the distribution and identify potential outliers.

### Detect Outliers Using the IQR Method
The Interquartile Range (IQR) method is used to detect outliers. The function calculates the first (Q1) and third (Q3) quartiles for each numeric column, determines the IQR, and sets the lower and upper bounds for outlier detection.

---

## Model Development

### Multi-Class Classification
1. **Random Forest Classifier**:
   - A `RandomForestClassifier` is used to fit the scaled training data.
   - Hyperparameter tuning is performed using `GridSearchCV`.
   - The best model from `GridSearchCV` is used for evaluation.
2. **Model Evaluation**:
   - Accuracy, classification report, and confusion matrix are calculated for both the validation and test sets.

### Binary Classification
1. **Random Forest Classifier**:
   - A separate `RandomForestClassifier` is trained on the scaled binary dataset.
2. **Model Evaluation**:
   - Accuracy, classification report, and confusion matrix are calculated for the binary validation and test sets.

---

## Feature Importance
The feature importances are calculated for the Random Forest model, providing insights into which features are most significant for the classification tasks.

---

## Requirements

- Python 3
- Required Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `gdown`

---

## Usage

1. **Download and Prepare the Dataset**:
   - Download the UNR-IDD dataset from the [official website](https://www.tapadhirdas.com/unr-idd-dataset) and place it in the appropriate directory.
2. **Install Required Libraries**:
   - Make sure all the required libraries are installed.
3. **Run the Notebook**:
   - Execute the notebook cells step-by-step to load the data, perform preprocessing, train models, and evaluate their performance.

---

## Results and Analysis

- **Multi-Class Classification**:
  - The model performance metrics including accuracy, precision, recall, and F1-score are presented for each attack type.
- **Binary Classification**:
  - The model's ability to differentiate between normal traffic and attacks is measured using accuracy, confusion matrix, and classification report.

---

## Conclusion

The NIDS project successfully demonstrates the use of machine learning models for detecting various types of network attacks. The results show that with proper data preprocessing, feature scaling, and model tuning, effective intrusion detection can be achieved for both multi-class and binary classification tasks.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- The University of Nevada - Reno for providing the Intrusion Detection Dataset (UNR-IDD). The dataset is available for download from the [official website](https://www.tapadhirdas.com/unr-idd-dataset).
- Scikit-learn, Pandas, and Matplotlib libraries for their powerful data processing and visualization capabilities.
