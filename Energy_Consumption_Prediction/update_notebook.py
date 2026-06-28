import json
import nbformat
from nbclient import NotebookClient

# Load notebook
with open('Refined_XGBoost_Energy_Consumption.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# We want to insert the Pipeline imports at the top
for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        if 'from xgboost import XGBRegressor' in source and 'Pipeline' not in source:
            source += "\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
            cell.source = source
        
        # Replace model initialization
        if '# Initialize and train the XGBoost Regressor' in source:
            new_source = source.replace("xgb_model = XGBRegressor(", "xgb_model = Pipeline([\n    ('scaler', StandardScaler()),\n    ('xgb', XGBRegressor(")
            # Find where to close the Pipeline parenthesis
            # We know the original ends with ')\n\nxgb_model.fit' or similar
            # Wait, this might be fragile. Let's just rewrite the whole cell source if we detect this is the training cell
            cell.source = """# Initialize and train the XGBoost Regressor inside a Pipeline
xgb_model = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("--- XGBoost Performance ---")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"MAE:  {mae_xgb:.2f}")
print(f"R2 Score: {r2_xgb:.4f}")
"""
        
        # Update feature importances extraction to use the named step
        if 'xgb_model.feature_importances_' in source:
            cell.source = source.replace("xgb_model.feature_importances_", "xgb_model.named_steps['xgb'].feature_importances_")

# Execute the notebook to update outputs and generate the model
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

# Save the updated notebook
with open('Refined_XGBoost_Energy_Consumption.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated and executed successfully.")
