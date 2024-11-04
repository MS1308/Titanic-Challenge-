# Titanic Survival Prediction Model

This project aims to predict the survival of passengers on the Titanic using machine learning techniques, specifically with a Random Forest Classifier. The model is trained on the Titanic dataset from Kaggle and includes data preprocessing, feature engineering, and validation.

## Project Overview

1. **Data Loading**: The dataset is loaded and explored to understand the structure and content of the data.
2. **Data Preprocessing**:
   - Missing values in the `Age` column are filled with the median age.
   - Missing values in the `Embarked` column are filled with the most common embarkation point.
   - The `Cabin` column is dropped due to a high percentage of missing values.
3. **Feature Engineering**:
   - Selected relevant features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
   - Categorical variables are one-hot encoded.
   - Numerical variables are scaled for improved model performance.
4. **Model Selection**: A Random Forest Classifier is chosen for its ability to handle both categorical and numerical data and its robustness in classification tasks.
5. **Pipeline and Training**:
   - A pipeline is created for seamless data transformation and model fitting.
   - The data is split into training and validation sets for performance evaluation.
6. **Evaluation**: The model's accuracy on the validation set is calculated.
7. **Prediction and Submission**:
   - The model is used to make predictions on the test data.
   - Results are saved in a CSV file ready for Kaggle submission.

## Files in the Project

- **train.csv**: Training dataset with survival outcomes for model training.
- **test.csv**: Test dataset without survival outcomes for model predictions.
- **submission.csv**: Generated file with passenger IDs and predicted survival outcomes for Kaggle submission.
- **Titanic_Model.ipynb**: Jupyter notebook containing the model implementation.

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn

## Usage

1. **Install required libraries**:
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. **Run the model**:
   Execute the Jupyter notebook or Python script to load data, preprocess, train, and generate the predictions.
3. **Generate the submission file**:
   After training, the model saves the predictions in `submission.csv` for direct upload to Kaggle.

## Model Performance

- Validation Accuracy: ~81.56%

## Future Improvements

- Tune model hyperparameters for improved accuracy.
- Explore additional feature engineering techniques.
- Experiment with other machine learning algorithms for comparison.

## Acknowledgments

This project uses the Titanic dataset provided by [Kaggle](https://www.kaggle.com/c/titanic).  
