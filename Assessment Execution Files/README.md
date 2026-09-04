# Customer Churn Prediction and Segmentation System

## Project Description

This project uses Machine Learning to help a telecom-style business understand
its customers in two complementary ways:

1. **Churn Prediction** — predicts whether a customer is likely to stop using
   the company's services (churn), based on their account and usage details.
2. **Customer Segmentation** — groups customers into meaningful segments using
   clustering, based on their tenure and billing behavior, so that different
   customer groups can be targeted with different retention strategies.

The system is built entirely in Python using standard, well-known Machine
Learning libraries, making it easy to understand, run, and explain.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Machine Learning Algorithms

### Logistic Regression
A simple, interpretable linear model used as a baseline classifier for churn
prediction. It estimates the probability that a customer will churn based on
a weighted combination of their features.

### Decision Tree Classifier
A tree-based model that splits customers into groups using simple if/else
rules on their features. Easy to visualize and interpret, but prone to
overfitting on its own.

### Random Forest Classifier
An ensemble of many decision trees whose predictions are averaged together.
It typically reduces overfitting compared to a single decision tree and
often improves prediction accuracy.

### K-Means Clustering
An unsupervised learning algorithm used for customer segmentation. It groups
customers into `k` clusters based on similarity in tenure, monthly charges,
and total charges — without using the churn label at all.

## Project Workflow

```
Data Collection --> Data Preprocessing --> Churn Prediction --> Model Evaluation --> Customer Segmentation --> Visualization
```

1. **Data Collection** — Load the Telco Customer Churn dataset from `data/`.
2. **Data Preprocessing** — Clean missing values, drop unnecessary columns,
   encode categorical features, and scale numerical features.
3. **Churn Prediction** — Train Logistic Regression, Decision Tree, and
   Random Forest classifiers on the processed data.
4. **Model Evaluation** — Compare all models using Accuracy, Precision,
   Recall, F1 Score, and Confusion Matrix; automatically select and save the
   best-performing model.
5. **Customer Segmentation** — Use the Elbow Method to choose a suitable
   number of clusters, then apply K-Means to assign each customer to a
   segment based on Tenure, Monthly Charges, and Total Charges.
6. **Visualization** — Generate charts for churn distribution, model
   comparison, the confusion matrix, the elbow curve, and the customer
   segments.

## Project Structure

```
MLA0201---FOML/
│
├── data/
│   ├── Telco-Customer-Churn.csv   # dataset used by the project
│   └── README.md                  # dataset details & download instructions
│
├── models/
│   └── best_model_*.joblib        # generated automatically when you run main.py
│
├── outputs/
│   ├── churn_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── elbow_curve.png
│   ├── customer_segments.png
│   ├── cluster_distribution.png
│   ├── segmented_customers.csv
│   ├── segment_profile.csv
│   └── model_performance_summary.csv
│   (all generated automatically when you run main.py)
│
├── src/
│   ├── data_preprocessing.py      # cleaning, encoding, scaling
│   ├── churn_prediction.py        # model training, evaluation, saving
│   ├── customer_segmentation.py   # K-Means clustering pipeline
│   └── visualization.py           # all chart generation
│
├── main.py                        # single entry point that runs everything
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/192572037simats-tech/MLA0201---FOML.git
cd MLA0201---FOML
pip install -r requirements.txt
python main.py
```

The dataset is already included in `data/Telco-Customer-Churn.csv`, so the
project runs immediately with no manual setup. See `data/README.md` if you
ever want to re-download it.

## Output

Running `python main.py` will:

- Print preprocessing details and model training progress to the console.
- Print Accuracy, Precision, Recall, and F1 Score for every model.
- Save the best-performing churn prediction model to `models/` as a
  `.joblib` file.
- Save customer segmentation results (`segmented_customers.csv` and
  `segment_profile.csv`) to `outputs/`.
- Save a model comparison summary (`model_performance_summary.csv`) to
  `outputs/`.
- Generate and save six charts to `outputs/`:
  - Churn distribution
  - Model comparison
  - Confusion matrix (best model)
  - Elbow method curve
  - Customer segments scatter plot
  - Cluster distribution

## Notes

- All model files and generated charts/CSVs are excluded from version
  control via `.gitignore` since they are reproducible by simply running
  `python main.py`.
- Random seeds are fixed (`random_state=42`) so results are reproducible
  between runs.
