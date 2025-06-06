import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load data
data = pd.read_csv('SwedishMotor.csv')

# Features and target
x = data.drop(columns='Payment', axis=1)
y = data['Payment']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Train models
model_random = RandomForestRegressor()
random = model_random.fit(x_train, y_train)

model_linear = LinearRegression()
linear = model_linear.fit(x_train, y_train)

model_extral = ExtraTreesRegressor()
extral = model_extral.fit(x_train, y_train)

model_gradient = GradientBoostingRegressor()
gradient = model_gradient.fit(x_train, y_train)

model_decision = DecisionTreeRegressor()
decision = model_decision.fit(x_train, y_train)

# Predict on test set
predict_random = random.predict(x_test)
predict_linear = linear.predict(x_test)
predict_extral = extral.predict(x_test)
predict_gradient = gradient.predict(x_test)
predict_decision = decision.predict(x_test)

# Create comparison dataframe
df1 = pd.DataFrame({
    'actual': y_test.reset_index(drop=True),
    'random': predict_random,
    'linear': predict_linear,
    'extral': predict_extral,
    'gradient': predict_gradient,
    'decision': predict_decision
})

# Plot predictions vs actual for first 11 samples
plt.subplot(221)
plt.plot(df1['actual'].iloc[0:11], label='Actual')
plt.plot(df1['random'].iloc[0:11], label='random')
plt.legend()

plt.subplot(222)
plt.plot(df1['actual'].iloc[0:11], label='Actual')
plt.plot(df1['linear'].iloc[0:11], label='linear')
plt.legend()

plt.subplot(223)
plt.plot(df1['actual'].iloc[0:11], label='Actual')
plt.plot(df1['extral'].iloc[0:11], label='extral')
plt.legend()

plt.subplot(224)
plt.plot(df1['actual'].iloc[0:11], label='Actual')
plt.plot(df1['gradient'].iloc[0:11], label='gradient')
plt.legend()

plt.tight_layout()
plt.show()

# Plot decision tree separately for clarity
plt.figure()
plt.plot(df1['actual'].iloc[0:11], label='Actual')
plt.plot(df1['decision'].iloc[0:11], label='decision')
plt.legend()
plt.title('Decision Tree Predictions vs Actual')
plt.show()

# Calculate R2 scores (correct order: y_test first, predictions second)
random_r2 = r2_score(y_test, predict_random)
linear_r2 = r2_score(y_test, predict_linear)
extral_r2 = r2_score(y_test, predict_extral)
gradient_r2 = r2_score(y_test, predict_gradient)
decision_r2 = r2_score(y_test, predict_decision)

print('R2 Scores:', random_r2, linear_r2, extral_r2, gradient_r2, decision_r2)

# Calculate Mean Absolute Error (MAE)
random_mae = mean_absolute_error(y_test, predict_random)
linear_mae = mean_absolute_error(y_test, predict_linear)
extral_mae = mean_absolute_error(y_test, predict_extral)
gradient_mae = mean_absolute_error(y_test, predict_gradient)
decision_mae = mean_absolute_error(y_test, predict_decision)

print('Mean Absolute Errors:', random_mae, linear_mae, extral_mae, gradient_mae, decision_mae)

# Predict new example
new = {'Kilometres': 1, 'Zone': 1, 'Bonus': 1, 'Make': 5, 'Insured': 191.01, 'Claims': 40}
df = pd.DataFrame(new, index=[0])

new_linear_pred = linear.predict(df)
new_random_pred = random.predict(df)
new_gradient_pred = gradient.predict(df)
new_decision_pred = decision.predict(df)
new_extral_pred = extral.predict(df)

print('New car price predictions are:', 
      new_linear_pred[0], new_random_pred[0], new_gradient_pred[0], new_decision_pred[0], new_extral_pred[0])

# Save models
joblib.dump(extral, 'model_joblib_extral')
joblib.dump(decision, 'model_joblib_decision')

