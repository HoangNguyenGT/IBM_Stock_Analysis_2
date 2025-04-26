# compare_ibm_multivariate.py - with Optuna hyperparameter tuning, Prophet, and build_calendar_exog
# ----------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import warnings, numpy as np, pandas as pd
import traceback
import matplotlib.pyplot as plt
import optuna
import random # Added for reproducibility

# Direct statsmodels import instead of sktime wrappers
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet # Added for Prophet model
# ----------------------------------------------------------------------
# Set random seeds for reproducibility
random.seed(123)
np.random.seed(123)
# ----------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Set pandas display options for debugging
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)

CSV = Path("market_data_rebased.csv")
if not CSV.exists():
    raise FileNotFoundError("market_data_rebased.csv not found – run fetch script first.")

# --- Utility function for MAPE calculation ---
def calculate_mape(y_true, y_pred):
    """Calculates MAPE safely, handling potential zeros in y_true."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-9 # Avoid division by zero or near-zero
    if np.sum(mask) == 0: return np.inf
    # Ensure y_pred is aligned and has same length for masking
    if len(y_pred) != len(y_true):
         print(f"Warning: Length mismatch in MAPE calculation. y_true={len(y_true)}, y_pred={len(y_pred)}")
         # Attempt to align based on index if possible
         if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
             common_index = y_true.index.intersection(y_pred.index)
             if len(common_index) == 0: return np.inf # No overlap
             y_true_common = y_true.loc[common_index]
             y_pred_common = y_pred.loc[common_index]
             mask_common = np.abs(y_true_common) > 1e-9
             if np.sum(mask_common) == 0: return np.inf
             return np.mean(np.abs((y_true_common[mask_common] - y_pred_common[mask_common]) / y_true_common[mask_common])) * 100
         else:
            # Try aligning by assuming sequential data if lengths differ
            min_len = min(len(y_true), len(y_pred))
            y_true_aligned = y_true[:min_len]
            y_pred_aligned = y_pred[:min_len]
            mask_aligned = np.abs(y_true_aligned) > 1e-9
            if np.sum(mask_aligned) == 0: return np.inf
            return np.mean(np.abs((y_true_aligned[mask_aligned] - y_pred_aligned[mask_aligned]) / y_true_aligned[mask_aligned])) * 100

    # If lengths match initially
    y_pred_masked = y_pred[mask]
    if len(y_true[mask]) != len(y_pred_masked):
         # This case should ideally not happen if lengths matched initially, but handle defensively
         print(f"Warning: Length mismatch after masking (initial lengths matched). y_true[mask]={len(y_true[mask])}, y_pred_masked={len(y_pred_masked)}")
         min_len = min(len(y_true[mask]), len(y_pred_masked))
         y_true_m_aligned = y_true[mask][:min_len]
         y_pred_m_aligned = y_pred_masked[:min_len]
         # Recalculate mask on the aligned subset
         mask_final = np.abs(y_true_m_aligned) > 1e-9
         if np.sum(mask_final) == 0: return np.inf
         return np.mean(np.abs((y_true_m_aligned[mask_final] - y_pred_m_aligned[mask_final]) / y_true_m_aligned[mask_final])) * 100

    return np.mean(np.abs((y_true[mask] - y_pred_masked) / y_true[mask])) * 100

# --- Utility function for data verification ---
def verify_data(df, name="DataFrame"):
    """Checks a DataFrame for NaNs or Infs and raises an error if found."""
    if df.isna().any().any():
        print(f"NaNs found in {name}:")
        print(df.isna().sum())
        raise ValueError(f"{name} contains NaNs after cleaning!")
    # Select only numeric columns before checking for inf
    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty and np.isinf(numeric_cols.values).any():
        print(f"Infs found in numeric columns of {name}:")
        # Identify columns with infs within the numeric subset
        inf_cols = numeric_cols.columns[np.isinf(numeric_cols.values).any(axis=0)]
        print(inf_cols)
        raise ValueError(f"{name} contains Infs after cleaning!")
    print(f"{name} verified clean (no NaNs or Infs).")

# --- Function to build exogenous variables ---
def build_calendar_exog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds exogenous variables including calendar dummies from a DataFrame.

    Args:
        df: Daily-frequency DataFrame with DateTimeIndex and columns
            including IBM, SP500, GOLD, BOND (prices or returns).

    Returns:
        DataFrame X containing selected numeric columns and calendar dummies.
    """
    print(f"Initial df shape for build_calendar_exog: {df.shape}")
    # Filter rows: Keep only dates where both IBM and SP500 are non-missing
    filtered_df = df[df[["IBM", "SP500"]].notna().all(axis=1)].copy()
    print(f"Shape after filtering IBM/SP500 NaNs: {filtered_df.shape}")
    if filtered_df.empty:
        raise ValueError("DataFrame is empty after filtering for IBM/SP500 NaNs.")

    # Numeric exogenous block
    numeric_cols = ["SP500", "GOLD", "BOND"]
    numeric_exog = filtered_df[numeric_cols]
    print(f"Numeric exog shape: {numeric_exog.shape}")

    # Calendar dummies (built on the filtered index)
    filtered_index = filtered_df.index
    # Day of week dummies (Monday=0, Sunday=6) -> omit Monday (dow_0)
    dow = pd.get_dummies(filtered_index.dayofweek, prefix="dow", drop_first=False) # Keep all initially
    if 0 in dow.columns: # Check if Monday (dow_0) exists before dropping
        dow = dow.drop(columns=[0]) # Drop Monday as reference
    dow.index = filtered_index # Reassign index

    # Month end dummy
    m_end = pd.Series(filtered_index.is_month_end.astype(int), index=filtered_index, name="m_end")

    # Concatenate numeric and dummy blocks
    X = pd.concat([numeric_exog, dow, m_end], axis=1)

    # Ensure desired column order (adjust if dow columns differ)
    final_cols_order = numeric_cols + [f"dow_{i}" for i in range(1, 7) if f"dow_{i}" in X.columns] + ["m_end"]
    X = X[final_cols_order]

    print(f"Final X shape in build_calendar_exog: {X.shape}")
    # print("X head:\n", X.head()) # Optional verification
    return X


# 1 ────────────────────────────────────────────────────────────────────
#   Load data and prepare target/exogenous series
# ----------------------------------------------------------------------
df_raw = pd.read_csv(CSV, parse_dates=["Date"], index_col="Date")
print(f"Raw data shape: {df_raw.shape}")

# Select relevant columns
cols_to_use = ["IBM", "SP500", "GOLD", "BOND"]
df_subset = df_raw[cols_to_use].copy()

# Step 1: Replace 0 with NaN BEFORE ffill
df_subset.replace(0, np.nan, inplace=True)
# Step 2: Forward fill NaNs
df_subset.fillna(method='ffill', inplace=True)
# Step 3: Drop any remaining NaNs (e.g., if ffill couldn't fill initial NaNs)
df_subset.dropna(inplace=True)
print(f"Shape after initial fill/drop: {df_subset.shape}")
verify_data(df_subset, "df_subset after initial clean")

# Step 4: Build Exogenous Variables using the function (on price data)
# This also filters based on IBM/SP500 availability
X_built = build_calendar_exog(df_subset)
verify_data(X_built, "X_built (prices + dummies)")

# Step 5: Calculate Target (IBM Returns) using the original subset aligned to X_built's index
y_full = df_subset['IBM'].pct_change()
y_full = y_full.loc[X_built.index] # Align y with the filtered index of X
print(f"y_full shape after pct_change and align: {y_full.shape}")

# Step 6: Calculate Returns for Numeric Exogenous Variables
numeric_cols_for_returns = ["SP500", "GOLD", "BOND"]
X_numeric_returns = X_built[numeric_cols_for_returns].pct_change()
print(f"X_numeric_returns shape: {X_numeric_returns.shape}")

# Step 7: Combine Numeric Returns and Dummy Variables
X_dummies = X_built.drop(columns=numeric_cols_for_returns)
X_processed = pd.concat([X_numeric_returns, X_dummies], axis=1)
print(f"X_processed shape before dropna: {X_processed.shape}")

# Step 8: Align y and X by dropping NaNs from pct_change
common_index = y_full.index.intersection(X_processed.index)
y_aligned = y_full.loc[common_index].dropna()
X_aligned = X_processed.loc[common_index].dropna()
# Final alignment check
final_common_index = y_aligned.index.intersection(X_aligned.index)
y = y_aligned.loc[final_common_index]
X_final_no_const = X_aligned.loc[final_common_index]

print(f"Shape after pct_change NaN drop: y={y.shape}, X={X_final_no_const.shape}")
if not y.index.equals(X_final_no_const.index):
     raise ValueError("Index mismatch between y and X after final alignment!")

# Step 9: Ensure all columns are numeric for OLS and add constant
X_final_no_const = X_final_no_const.astype(np.float64) # Convert all columns to float
X = sm.add_constant(X_final_no_const, has_constant='add') # Use 'add' as we start without one

# Step 10: Final Verification
verify_data(y.to_frame(), "y (final)")
verify_data(X, "X (final with const)")
print("\nFinal Cleaned data shapes:")
print(f"Target shape: {y.shape}")
print(f"Exogenous features shape (with const): {X.shape}")
print(f"Final Exogenous columns: {X.columns.tolist()}")


# 2 ────────────────────────────────────────────────────────────────────
#   Train/validation/test split (using CLEANED & ALIGNED data)
# ----------------------------------------------------------------------
test_size = 21
val_size = 21
val_end = len(y) - test_size
train_end = val_end - val_size

# Split the final aligned y and X (with constant)
y_train = y[:train_end]
X_train = X[:train_end]
y_val = y[train_end:val_end]
X_val = X[train_end:val_end]
y_test = y[val_end:]
X_test = X[val_end:]

print("\nData splits:")
print(f"Training:    {len(y_train)} obs, X_train shape: {X_train.shape}")
print(f"Validation:  {len(y_val)} obs, X_val shape: {X_val.shape}")
print(f"Testing:     {len(y_test)} obs, X_test shape: {X_test.shape}")

# --- Final Verification of Splits ---
verify_data(y_train.to_frame(), "y_train")
verify_data(X_train, "X_train")
verify_data(y_val.to_frame(), "y_val")
verify_data(X_val, "X_val")
verify_data(y_test.to_frame(), "y_test")
verify_data(X_test, "X_test")
print("Data splits verified clean.")

# 2b ───────────────────────────────────────────────────────────────────
#   Prepare data specifically for Prophet
# ----------------------------------------------------------------------
# Prophet uses the numeric RETURN columns (not prices or dummies)
prophet_regressors = ['SP500', 'GOLD', 'BOND'] # These are now return columns in X

# Training data
df_train_prophet = X_train[prophet_regressors].copy()
df_train_prophet['y'] = y_train
df_train_prophet.reset_index(inplace=True)
df_train_prophet.rename(columns={'Date': 'ds'}, inplace=True)
verify_data(df_train_prophet, "df_train_prophet")

# Validation data
df_val_prophet = X_val[prophet_regressors].copy()
df_val_prophet['y'] = y_val
df_val_prophet.reset_index(inplace=True)
df_val_prophet.rename(columns={'Date': 'ds'}, inplace=True)
verify_data(df_val_prophet, "df_val_prophet")

# Test data
df_test_prophet = X_test[prophet_regressors].copy()
df_test_prophet['y'] = y_test
df_test_prophet.reset_index(inplace=True)
df_test_prophet.rename(columns={'Date': 'ds'}, inplace=True)
verify_data(df_test_prophet, "df_test_prophet")

print("\nProphet data shapes:")
print(f"Train: {df_train_prophet.shape}, Val: {df_val_prophet.shape}, Test: {df_test_prophet.shape}")


# 3 ────────────────────────────────────────────────────────────────────
#   Direct OLS (Market Model) benchmark
# ----------------------------------------------------------------------
print("\n>>> Training Market Model (Linear regression with ALL new features)")
results = []
try:
    # OLS now uses X_train which includes numeric returns and calendar dummies + const
    ols_model = sm.OLS(y_train, X_train)
    ols_model_fit = ols_model.fit()
    ols_predictions = ols_model_fit.predict(X_test)
    ols_mape = calculate_mape(y_test, ols_predictions)
    print(f"Market model MAPE: {ols_mape:.2f}%")
    results.append(("Market_Model", ols_mape))

    # Plotting moved to section 8
except Exception as e:
    print(f"Error with Market Model: {str(e)}")
    traceback.print_exc()
    results.append(("Market_Model", np.nan))
# Ensure ols_predictions exists even if model failed
ols_predictions = ols_predictions if 'ols_predictions' in locals() else pd.Series(np.nan, index=y_test.index)


# 4 ────────────────────────────────────────────────────────────────────
#   Optuna optimization for SARIMAX hyperparameters
# ----------------------------------------------------------------------
print("\n>>> Starting Optuna hyperparameter tuning for SARIMAX")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define SARIMAX regressors (numeric returns + constant, NO dummies as per prior request)
sarimax_regressors = ['const', 'SP500', 'GOLD', 'BOND']

def objective_sarimax(trial): # Renamed objective function
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 2)

    # Select ONLY the specified regressors for SARIMAX
    X_train_subset = X_train[sarimax_regressors]
    X_val_subset = X_val[sarimax_regressors]

    try:
        # Verify data just before fitting THIS trial
        verify_data(y_train.to_frame(), f"y_train (SARIMAX Trial {trial.number})")
        verify_data(X_train_subset, f"X_train_subset (SARIMAX Trial {trial.number})")

        model = SARIMAX(
            y_train, exog=X_train_subset, order=(p, d, q),
            enforce_stationarity=False, enforce_invertibility=False
        )
        model_fit = model.fit(disp=False, maxiter=100)

        # Verify forecast exog data
        verify_data(X_val_subset, f"X_val_subset (SARIMAX Trial {trial.number})")
        predictions = model_fit.forecast(steps=len(y_val), exog=X_val_subset)

        mape = calculate_mape(y_val, predictions)

        if np.isnan(mape) or np.isinf(mape):
             print(f"SARIMAX T{trial.number} MAPE is NaN/Inf")
             return 1000.0
        return mape
    except Exception as e:
        # print(f"SARIMAX T{trial.number} failed: {str(e)}") # Muted unless debugging
        return 1000.0

study_sarimax = optuna.create_study(direction='minimize') # Renamed study
n_trials_sarimax = 30 # Use separate trial count if desired
print(f"Running Optuna for SARIMAX with {n_trials_sarimax} trials...")
study_sarimax.optimize(objective_sarimax, n_trials=n_trials_sarimax, show_progress_bar=True) # Use renamed objective

print("\nBest SARIMAX parameters found by Optuna:")
try:
    best_params_sarimax = study_sarimax.best_params # Renamed best_params
    best_value_sarimax = study_sarimax.best_value # Renamed best_value
    print(f"Best validation MAPE: {best_value_sarimax:.4f}%")
    print(f"Parameters: {best_params_sarimax}")
except optuna.exceptions.OptunaError:
    print("Optuna study for SARIMAX finished without finding any successful trials.")
    best_params_sarimax = None

# 5 ────────────────────────────────────────────────────────────────────
#   Train final SARIMAX model with best parameters (if found)
# ----------------------------------------------------------------------
sarimax_predictions = pd.Series(np.nan, index=y_test.index) # Initialize with NaNs
if best_params_sarimax: # Check renamed variable
    print("\n>>> Training final SARIMAX model with optimal parameters")
    p, d, q = best_params_sarimax['p'], best_params_sarimax['d'], best_params_sarimax['q'] # Use renamed variable

    # Use ONLY the specified regressors for SARIMAX
    X_cols_sarimax = sarimax_regressors
    print(f"Selected features for SARIMAX: {X_cols_sarimax}")

    # Combine cleaned training and validation data, selecting SARIMAX cols
    y_train_full = pd.concat([y_train, y_val])
    X_train_full = pd.concat([X_train, X_val])[X_cols_sarimax]
    X_test_subset = X_test[X_cols_sarimax]

    # Final verification before final model fit
    verify_data(y_train_full.to_frame(), "y_train_full (SARIMAX)")
    verify_data(X_train_full, "X_train_full (SARIMAX)")
    verify_data(X_test_subset, "X_test_subset (SARIMAX)")

    try:
        final_model = SARIMAX(
            y_train_full, exog=X_train_full, order=(p, d, q),
            enforce_stationarity=False, enforce_invertibility=False
        )
        final_model_fit = final_model.fit(disp=False)
        sarimax_predictions = final_model_fit.forecast(steps=len(y_test), exog=X_test_subset) # Overwrite if successful
        sarimax_mape = calculate_mape(y_test, sarimax_predictions)
        print(f"SARIMAX test MAPE: {sarimax_mape:.2f}%")
        results.append(("SARIMAX", sarimax_mape))

        # Plotting moved to section 8
    except Exception as e:
        print(f"Error with final SARIMAX model: {str(e)}")
        traceback.print_exc()
        results.append(("SARIMAX", np.nan))
else:
    print("\nSkipping final SARIMAX model training as Optuna did not find valid parameters.")
    if "SARIMAX" not in [r[0] for r in results]:
         results.append(("SARIMAX", np.nan))


# 6 ────────────────────────────────────────────────────────────────────
#   Optuna optimization for Prophet hyperparameters
# ----------------------------------------------------------------------
print("\n>>> Starting Optuna hyperparameter tuning for Prophet")
# Suppress Prophet informational messages during tuning
# import logging # Already imported earlier if needed, but good practice to ensure
# logging.getLogger('prophet').setLevel(logging.ERROR)
# logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def objective_prophet(trial):
    # Suggest hyperparameters
    changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
    regressor_prior_scale = trial.suggest_float('regressor_prior_scale', 0.01, 10, log=True)
    # seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True) # Optional: tune seasonality if needed

    try:
        # Instantiate Prophet model
        m = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            # seasonality_prior_scale=seasonality_prior_scale, # Uncomment if tuning seasonality
            daily_seasonality=False, # Assuming daily returns don't have strong daily pattern
            weekly_seasonality=False, # Assuming daily returns don't have strong weekly pattern
            yearly_seasonality=False # Assuming daily returns don't have strong yearly pattern
        )

        # Add regressors (numeric returns)
        for regressor in prophet_regressors: # Uses ['SP500', 'GOLD', 'BOND']
            m.add_regressor(regressor, prior_scale=regressor_prior_scale, mode='additive') # Keep mode additive for returns

        # Fit model on training data (Prophet format)
        m.fit(df_train_prophet[['ds', 'y'] + prophet_regressors])

        # Create future dataframe for validation period
        # Need to use the validation data prepared for Prophet
        future_val = df_val_prophet[['ds'] + prophet_regressors].copy()
        verify_data(future_val, f"future_val (Prophet Trial {trial.number})") # Verify before predict

        # Predict
        forecast_val = m.predict(future_val)

        # Extract predictions and calculate MAPE
        y_pred_val = forecast_val['yhat']
        y_true_val = df_val_prophet['y'] # Get actuals from the validation Prophet df
        mape = calculate_mape(y_true_val.values, y_pred_val.values) # Use .values to avoid index issues

        if np.isnan(mape) or np.isinf(mape):
             print(f"Prophet T{trial.number} MAPE is NaN/Inf")
             return 1000.0
        return mape

    except Exception as e:
        # print(f"Prophet T{trial.number} failed: {str(e)}") # Muted unless debugging
        return 1000.0

study_prophet = optuna.create_study(direction='minimize')
n_trials_prophet = 30 # Can adjust number of trials for Prophet
print(f"Running Optuna for Prophet with {n_trials_prophet} trials...")
study_prophet.optimize(objective_prophet, n_trials=n_trials_prophet, show_progress_bar=True)

print("\nBest Prophet parameters found by Optuna:")
try:
    best_params_prophet = study_prophet.best_params
    best_value_prophet = study_prophet.best_value
    print(f"Best validation MAPE: {best_value_prophet:.4f}%")
    print(f"Parameters: {best_params_prophet}")
except optuna.exceptions.OptunaError:
    print("Optuna study for Prophet finished without finding any successful trials.")
    best_params_prophet = None

# 7 ────────────────────────────────────────────────────────────────────
#   Train final Prophet model with best parameters (if found)
# ----------------------------------------------------------------------
prophet_predictions = pd.Series(np.nan, index=y_test.index) # Initialize with NaNs
if best_params_prophet:
    print("\n>>> Training final Prophet model with optimal parameters")
    changepoint_prior_scale = best_params_prophet['changepoint_prior_scale']
    regressor_prior_scale = best_params_prophet['regressor_prior_scale']
    # seasonality_prior_scale = best_params_prophet.get('seasonality_prior_scale', 10.0) # Use default if not tuned

    # Combine training and validation data for Prophet
    df_train_full_prophet = pd.concat([df_train_prophet, df_val_prophet], ignore_index=True)
    verify_data(df_train_full_prophet, "df_train_full_prophet")

    try:
        # Instantiate final Prophet model
        final_model_prophet = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            # seasonality_prior_scale=seasonality_prior_scale, # Uncomment if tuned
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False
        )

        # Add regressors (numeric returns)
        for regressor in prophet_regressors: # Uses ['SP500', 'GOLD', 'BOND']
            final_model_prophet.add_regressor(regressor, prior_scale=regressor_prior_scale, mode='additive')

        # Fit model on combined training + validation data
        final_model_prophet.fit(df_train_full_prophet[['ds', 'y'] + prophet_regressors])

        # Prepare the test dataframe for prediction (already contains future dates and regressors)
        df_predict_prophet = df_test_prophet[['ds'] + prophet_regressors].copy()
        verify_data(df_predict_prophet, "df_predict_prophet (Final Prophet)") # Verify before predict

        # Predict on test data using the prepared dataframe
        forecast_test = final_model_prophet.predict(df_predict_prophet)

        # Extract predictions and calculate MAPE
        prophet_predictions = forecast_test['yhat'] # Overwrite if successful
        prophet_mape = calculate_mape(df_test_prophet['y'].values, prophet_predictions.values) # Use .values
        print(f"Prophet test MAPE: {prophet_mape:.2f}%")
        results.append(("Prophet", prophet_mape))
        # Ensure prophet_predictions has the correct index for plotting
        prophet_predictions.index = y_test.index

    except Exception as e:
        print(f"Error with final Prophet model: {str(e)}")
        traceback.print_exc()
        results.append(("Prophet", np.nan))
else:
    print("\nSkipping final Prophet model training as Optuna did not find valid parameters.")
    if "Prophet" not in [r[0] for r in results]:
         results.append(("Prophet", np.nan))


# 8 ────────────────────────────────────────────────────────────────────
#   Results table and comparison Plots
# ----------------------------------------------------------------------
if results:
    # Ensure all prediction series exist, even if models failed
    ols_predictions = ols_predictions if 'ols_predictions' in locals() else pd.Series(np.nan, index=y_test.index)
    sarimax_predictions = sarimax_predictions if 'sarimax_predictions' in locals() else pd.Series(np.nan, index=y_test.index)
    prophet_predictions = prophet_predictions if 'prophet_predictions' in locals() else pd.Series(np.nan, index=y_test.index)

    valid_results = [(model, mape) for model, mape in results if pd.notna(mape) and not np.isinf(mape)]

    if valid_results:
        leader = pd.DataFrame(valid_results, columns=["model", "mape"]).set_index("model").sort_values("mape")
        print("\n=== Model performance (lower MAPE better) ===")
        print(leader)
        leader.to_csv("model_performance.csv")
        print("Results saved to 'model_performance.csv'")

        full_results = pd.DataFrame(results, columns=["model", "mape"]).set_index("model")
        print("\n=== Full results (including failed models/NaN MAPE) ===")
        print(full_results)
        full_results.to_csv("full_results.csv")

        # --- Plot 1: MAPE Bar Chart ---
        if not leader.empty:
            plt.figure(figsize=(8, 5))
            # Define consistent colors for models
            model_colors = {'Market_Model': 'green', 'SARIMAX': 'red', 'Prophet': 'purple'}
            bar_colors = [model_colors.get(model, 'gray') for model in leader.index] # Use gray for unexpected models
            leader['mape'].plot(kind='bar', color=bar_colors)
            plt.title('Model Comparison: Test Set MAPE')
            plt.ylabel('MAPE (%)')
            plt.xlabel('Model')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('mape_comparison_bar_chart.png')
            print("\nMAPE comparison bar chart saved to 'mape_comparison_bar_chart.png'")
        else:
            print("\nNo valid model results to plot MAPE bar chart.")


        # --- Plot 2: Combined Forecast Line Plot ---
        plt.figure(figsize=(14, 7))
        # Plot actuals as blue dots
        plt.plot(y_test.index, y_test, 'bo', label='Actual IBM Returns', markersize=4, alpha=0.6)

        # Determine best model for thicker line (only if leader is not empty)
        best_model_name = leader.index[0] if not leader.empty else None
        model_plot_params = {
            'Market_Model': {'color': 'green', 'linestyle': '--', 'alpha': 0.8},
            'SARIMAX': {'color': 'red', 'linestyle': ':', 'alpha': 0.8},
            'Prophet': {'color': 'purple', 'linestyle': '-.', 'alpha': 0.8} # Consistent Prophet color
        }

        # Plot Market Model if successful
        if 'Market_Model' in full_results.index and pd.notna(full_results.loc['Market_Model', 'mape']):
            model_name = 'Market_Model'
            mape_val = full_results.loc[model_name, 'mape']
            lw = 2.5 if model_name == best_model_name else 1.5 # Thicker line if best
            plt.plot(y_test.index, ols_predictions, label=f'{model_name} (MAPE: {mape_val:.2f}%)', linewidth=lw, **model_plot_params[model_name])

        # Plot SARIMAX if successful
        if 'SARIMAX' in full_results.index and pd.notna(full_results.loc['SARIMAX', 'mape']):
            model_name = 'SARIMAX'
            mape_val = full_results.loc[model_name, 'mape']
            lw = 2.5 if model_name == best_model_name else 1.5 # Thicker line if best
            plt.plot(y_test.index, sarimax_predictions, label=f'{model_name} (MAPE: {mape_val:.2f}%)', linewidth=lw, **model_plot_params[model_name])

        # Plot Prophet if successful
        if 'Prophet' in full_results.index and pd.notna(full_results.loc['Prophet', 'mape']):
             model_name = 'Prophet'
             mape_val = full_results.loc[model_name, 'mape']
             lw = 2.5 if model_name == best_model_name else 1.5 # Thicker line if best
             # Ensure prophet_predictions index matches y_test index if not already done
             if not prophet_predictions.index.equals(y_test.index):
                 prophet_predictions.index = y_test.index
             plt.plot(y_test.index, prophet_predictions, label=f'{model_name} (MAPE: {mape_val:.2f}%)', linewidth=lw, **model_plot_params[model_name])

        plt.title('IBM Returns: Actual vs. Forecast Comparison (Test Set)')
        plt.xlabel('Date')
        plt.ylabel('Daily Return')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('combined_forecast_comparison.png')
        print("\nCombined forecast plot saved to 'combined_forecast_comparison.png'")


        # --- Optuna Analysis (SARIMAX) ---
        if 'study_sarimax' in locals() and hasattr(study_sarimax, 'best_trial') and study_sarimax.best_trial: # Check renamed study
            print("\n--- SARIMAX Optuna Analysis ---")
            try:
                importance = optuna.importance.get_param_importances(study_sarimax) # Use renamed study
                importance_df = pd.DataFrame(list(importance.items()), columns=['SARIMAX Parameter', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=False)
                importance_df.to_csv('sarimax_parameter_importance.csv', index=False) # Renamed file
                print("SARIMAX Parameter importance saved to 'sarimax_parameter_importance.csv'")

                try:
                    fig = optuna.visualization.plot_param_importances(study_sarimax) # Use renamed study
                    fig.write_image("optuna_sarimax_parameter_importance.png") # Renamed file
                    print("Optuna SARIMAX parameter importance plot saved to 'optuna_sarimax_parameter_importance.png'")
                except ImportError:
                     print("Plotly/Kaleido not installed, skipping Optuna SARIMAX visualization.")
                except Exception as e:
                    print(f"Could not create Optuna SARIMAX visualization: {str(e)}")
            except Exception as e:
                print(f"Error analyzing SARIMAX Optuna results: {str(e)}")
        else:
             print("\nSARIMAX Optuna study did not complete successfully, skipping results analysis.")

        # --- Optuna Analysis (Prophet) ---
        if 'study_prophet' in locals() and hasattr(study_prophet, 'best_trial') and study_prophet.best_trial:
            print("\n--- Prophet Optuna Analysis ---")
            try:
                importance_prophet = optuna.importance.get_param_importances(study_prophet)
                importance_df_prophet = pd.DataFrame(list(importance_prophet.items()), columns=['Prophet Parameter', 'Importance'])
                importance_df_prophet = importance_df_prophet.sort_values('Importance', ascending=False)
                importance_df_prophet.to_csv('prophet_parameter_importance.csv', index=False)
                print("Prophet Parameter importance saved to 'prophet_parameter_importance.csv'")

                try:
                    fig_prophet = optuna.visualization.plot_param_importances(study_prophet)
                    fig_prophet.write_image("optuna_prophet_parameter_importance.png")
                    print("Optuna Prophet parameter importance plot saved to 'optuna_prophet_parameter_importance.png'")
                except ImportError:
                     print("Plotly/Kaleido not installed, skipping Optuna Prophet visualization.")
                except Exception as e:
                    print(f"Could not create Optuna Prophet visualization: {str(e)}")
            except Exception as e:
                print(f"Error analyzing Prophet Optuna results: {str(e)}")
        else:
             print("\nProphet Optuna study did not complete successfully, skipping results analysis.")

    else:
        print("\nNo models were successfully trained with valid MAPE.")
else:
    print("\nNo models were successfully trained.")
