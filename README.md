# Auto-mobile Price Prediction  
Predict car insurance payments using advanced regression models in Python.

# 🚗 Auto-mobile Price Prediction with Regression Models

This project applies multiple regression algorithms to predict **car insurance payments** based on vehicle and policy features from the Swedish Motor dataset.

📌 **Objective**

Build and evaluate machine learning models that estimate insurance payments accurately using historical data on car mileage, zone, bonus, make, insured amount, and claims.

🛠️ **Tools & Technologies**

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, joblib)  
- Regression Models: Random Forest, Linear Regression, Extra Trees, Gradient Boosting, Decision Tree  
- Model Evaluation Metrics: R² Score, Mean Absolute Error (MAE)  
- Data Source: SwedishMotor.csv  

📈 **Key Features**

- Data loading, preprocessing, and train-test split  
- Training and comparison of multiple regression models  
- Visualization of actual vs predicted insurance payments  
- Prediction on new example car data  
- Model persistence using joblib for future use  

🧪 **Results**

- **Best Model:** Extra Trees Regressor (example)  
- **Performance Metrics:** R² scores and MAE printed for all models  
- Visual plots illustrating model prediction accuracy  
- Saved models ready for deployment or further analysis  

📁 **Project Structure**

auto-mobile-price-prediction/
├── SwedishMotor.csv # Dataset file
├── auto_price_prediction.py # Main script with model training and evaluation
├── model_joblib_extral # Saved Extra Trees Regressor model
├── model_joblib_decision # Saved Decision Tree model
├── README.md # Project overview and documentation
└── requirements.txt # Python dependencies



🧑‍💻 **Author**

**Oyeyemi Olajuwon Usman**  
[GitHub Profile](https://github.com/oyeyemiolajuwon)

📄 **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

📊 **Preview**

Example plot comparing actual insurance payments vs predictions for the test set:

![Prediction Plot](charts/prediction_plot.png)  <!-- Optional: Add if you have this chart -->

---

💡 **How to Run**

1. Clone the repository  
2. Place the `SwedishMotor.csv` dataset in the root directory  
3. Install dependencies:

```bash
pip install -r requirements.txt
