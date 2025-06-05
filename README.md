🚗 Auto-mobile Insurance Payment Prediction with Machine Learning
This project demonstrates how to predict car insurance payments using multiple regression models trained on the SwedishMotor dataset. It covers the full ML pipeline: data preprocessing, training, evaluation, visualization, prediction on new data, and model saving.

📌 Objective
The goal is to predict the insurance payment amount based on several vehicle and policyholder features such as kilometers driven, geographic zone, bonus class, car make, insured amount, and claims history. This can help insurance companies estimate payments more accurately for underwriting and risk assessment.

🛠️ Tools & Technologies
Programming Language: Python

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

tkinter (used internally by matplotlib for plotting)

📊 Dataset
The dataset SwedishMotor.csv contains records with the following columns:

Column	Description
Kilometres	Kilometers driven by the vehicle
Zone	Geographic zone category
Bonus	Bonus class of the policyholder
Make	Car make encoded as a categorical int
Insured	Insured amount (monetary value)
Claims	Number of claims made
Payment	Insurance payment amount (target variable)

📁 Project Structure
bash
Copy
Edit
swedish-motor-insurance-prediction/
├── SwedishMotor.csv              # Dataset file
├── swedish_motor_prediction.py  # Main ML script (your script)
├── model_joblib_extral           # Saved Extra Trees Regressor model
├── model_joblib_decision         # Saved Decision Tree Regressor model
├── README.md                    # This documentation file
├── requirements.txt             # Python dependencies
🧠 Machine Learning Workflow
1. Data Loading and Inspection
Dataset loaded using pandas

Basic inspection for understanding the features and target variable

2. Train-Test Split
Split dataset into 80% training and 20% testing using train_test_split

Random state fixed for reproducibility

3. Model Training
Trained multiple regression models on the training data:

Random Forest Regressor

Linear Regression

Extra Trees Regressor

Gradient Boosting Regressor

Decision Tree Regressor

4. Prediction
Models predict insurance payments on the test set.

5. Evaluation Metrics
Models evaluated with:

R² Score (Coefficient of Determination)

Mean Absolute Error (MAE)

6. Visualization
Plotted actual vs predicted payment values for the first 11 test samples for each model, arranged in subplots for side-by-side comparison.

7. New Sample Prediction
Predicted insurance payment for a new hypothetical car policy with features:

python
Copy
Edit
{'Kilometres':1,'Zone':1,'Bonus':1,'Make':5,'Insured':191.01,'Claims':40}
Outputs predictions from all models are displayed.

8. Model Saving
Saved the Extra Trees and Decision Tree models using joblib for future inference.

📉 Visualization Overview
Top-left: Actual vs Random Forest predictions

Top-right: Actual vs Linear Regression predictions

Bottom-left: Actual vs Extra Trees predictions

Bottom-right: Actual vs Gradient Boosting predictions

Decision Tree plot also visualized separately

Each plot helps to compare model accuracy visually on the same test samples.

📈 Example Model Performance Output
plaintext
Copy
Edit
R² Scores (example):
Random Forest: 0.85
Linear Regression: 0.78
Extra Trees: 0.86
Gradient Boosting: 0.84
Decision Tree: 0.75

Mean Absolute Error (MAE):
Random Forest: 120.5
Linear Regression: 140.2
Extra Trees: 118.7
Gradient Boosting: 121.0
Decision Tree: 150.8

Predicted insurance payment for new sample:
Linear Regression: 1345.6
Random Forest: 1302.4
Gradient Boosting: 1318.7
Decision Tree: 1289.1
Extra Trees: 1320.5
(Note: Values will vary based on training and dataset.)

💾 Model Saving Code
python
Copy
Edit
import joblib

joblib.dump(extral, 'model_joblib_extral')
joblib.dump(decision, 'model_joblib_decision')
