# COVID-19 mRNA Vaccine Degradation Prediction

## Project Overview
This project aims to predict degradation rates of mRNA molecules at various sequence positions using advanced machine learning techniques. The dataset and problem are based on a Kaggle competition that addresses critical challenges in stabilizing COVID-19 mRNA vaccines. By accurately predicting the degradation rates, this project contributes to improving the shelf-life and efficacy of mRNA-based vaccines.

## Problem Statement
RNA molecules are prone to degradation, a critical limitation in mRNA vaccine technology. This project uses machine learning models to predict:
* **Reactivity**
* **Degradation under Magnesium at pH 10 (deg_Mg_pH10)**
* **Degradation under Magnesium at 50°C (deg_Mg_50C)**

The goal is to enable insights into degradation mechanisms and assist researchers in designing more stable mRNA sequences.

## Key Features

### 1. Data Preprocessing
* Merging additional features with training and test data
* Handling categorical features using Label Encoding
* Ensuring compatibility between train and test datasets

### 2. Machine Learning
* Training an **XGBoost Regressor** for multi-target regression
* Hyperparameter tuning using **GridSearchCV**
* Evaluating performance using **RMSE** (Root Mean Squared Error)

### 3. Visualization
* Scatter plots to compare predicted vs actual values for each target variable

### 4. Results
* Final predictions are exported as a CSV file (`test_predictions_xgb_tuned.csv`)

## Dataset Description

### Input Files
* `train.csv`: Contains training data with RNA sequences, structures, and target degradation rates
* `test.csv`: Test dataset with RNA sequences and structures (without ground truth targets)
* `df1.csv`: Supplementary data with additional features

### Columns of Interest

#### Features
* `sequence`: RNA sequence (A, G, U, C)
* `structure`: RNA structural information (., (, ))
* `predicted_loop_type`: Loop type information

#### Targets
* `reactivity`
* `deg_Mg_pH10`
* `deg_Mg_50C`

## Installation and Requirements

### Prerequisites
Ensure you have Python 3.8+ installed with the following libraries:
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib`

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/mrna-degradation-prediction.git
cd mrna-degradation-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place `train.csv`, `test.csv`, and `df1.csv` in the project directory.

## Usage

### Run the Model
1. **Preprocess the data**:
   * Merges training/test datasets with additional features
   * Encodes categorical features
   * Splits data into training and validation sets

2. **Train the model**:
   * Hyperparameter tuning using GridSearchCV
   * Trains the best XGBoost model on the dataset

3. **Predict and save results**:
   * Generates predictions for the test dataset
   * Saves predictions to `test_predictions_xgb_tuned.csv`

### Visualizations
* Use scatter plots to analyze predictions vs actual values for each target variable

### Example Commands
Run the main script:
```bash
python main.py
```

## Files and Directories
* **main.py**: Main script for preprocessing, model training, and evaluation
* **requirements.txt**: Lists required Python packages
* **train.csv**, **test.csv**, **df1.csv**: Input data files
* **test_predictions_xgb_tuned.csv**: Final prediction file for submission

## Results
The model achieves competitive RMSE scores for multi-target regression. Key performance highlights:
* **Reactivity RMSE**: [Reported value]
* **deg_Mg_pH10 RMSE**: [Reported value]
* **deg_Mg_50C RMSE**: [Reported value]

## Future Work
* Experiment with other regression models (e.g., Random Forest, Neural Networks)
* Explore feature engineering to improve model accuracy
* Incorporate additional datasets to enhance generalization
