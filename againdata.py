#%%
import numpy as np
import pandas as pd

data=pd.read_csv(r"C:\Users\HP\Desktop\ai salary prediction project\salary_prediction_dataset.csv")
# %%
data.head(10)
# %%
data.isnull().sum()
# %%
X=data[["Experience"]]
y=data["Salary"]

# %%
from sklearn.model_selection import train_test_split
# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# %%
X_test.to_csv(r"C:\Users\HP\Desktop\ai salary prediction project\salaryxtest")
X_train.to_csv(r"C:\Users\HP\Desktop\ai salary prediction project\salaryxtrain")
y_test.to_csv(r"C:\Users\HP\Desktop\ai salary prediction project\salaryytest")
y_train.to_csv(r"C:\Users\HP\Desktop\ai salary prediction project\salaryytrain")

# %%
from sklearn.dummy import DummyRegressor
# %%
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
# %%

# %%

# %%

# %%
# %%
# %%

# %%
# %%

import pandas as pd
import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Identify columns
categorical_cols = ["Experience"]

# 2. Create preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("encoding",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_cols)
    ],
    remainder="passthrough"
)

# 3. Define models
models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Median Baseline": DummyRegressor(strategy="median"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

# 4. Train, cross-validate, and evaluate
results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Cross-validation on training data
    cv_mae = -cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    ).mean()

    # Train on the full training split
    pipeline.fit(X_train, y_train)

    # Predict test data
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "CV_MAE": cv_mae,
        "Test_MAE": mae,
        "Test_MSE": mse,
        "Test_RMSE": rmse,
        "Test_R2": r2
    })

    trained_models[name] = pipeline

# 5. Compare all models
results_df = pd.DataFrame(results)
print(results_df.round(2).to_string(index=False))

# 6. Select the model with lowest cross-validation MAE
best_name = results_df.loc[
    results_df["CV_MAE"].idxmin(), "Model"
]
best_model = trained_models[best_name]

print("\nBest model by CV MAE:", best_name)

# 7. Display actual and predicted salaries
comparison = pd.DataFrame({
    "Actual_Salary": y_test.to_numpy(),
    "Predicted_Salary": best_model.predict(X_test)
})

print("\nActual vs Predicted:")
print(comparison.head(10).round(2))

# 8. Save the selected model and its preprocessing
import joblib

joblib.dump(best_model, "salary_prediction_model.pkl")
print("\nModel saved successfully.")

# %%
