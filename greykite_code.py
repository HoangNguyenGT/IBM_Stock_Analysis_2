from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig, ModelComponentsParam
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
import pandas as pd 

df = pd.read_csv('market_data_rebased.csv')
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date")
time_col = df.columns[0]
asset_cols = df.columns[1:len(df.columns)]
regressor_cols = []
df.head()

financial_crisis1 = pd.to_datetime("2008-09-25")
financial_crisis2 = pd.to_datetime("2008-11-25")
covid_crisis1 = pd.to_datetime("2020-03-06")
covid_crisis2 = pd.to_datetime("2020-05-06")
tarrifs_crisis1 = pd.to_datetime("2025-04-02")

crisis_mask = (df["Date"] >= financial_crisis1) & (df["Date"] < financial_crisis2)
crisis_mask = (crisis_mask) | ((df["Date"] >= covid_crisis1) & (df["Date"] < covid_crisis2))
crisis_mask = (crisis_mask) | ((df["Date"] >= tarrifs_crisis1))

df.loc[crisis_mask, "crisis"] = 1
df.loc[~crisis_mask, "crisis"] = 0
df.tail(26)


## BOND and GOLD add very little predictive power to the IBM predictibility. 
config = ForecastConfig(
     metadata_param=MetadataParam(time_col=time_col, 
                                  value_col="IBM", 
                                  train_end_date = "2025-03-26" ),  # Column names in `df`
     model_template=ModelTemplateEnum.SILVERKITE.name,  # SILVERKITE model configuration
     
     forecast_horizon=21,   # Forecasts 24 steps ahead
     coverage=0.95,         # 95% prediction intervals
     model_components_param=ModelComponentsParam(
         regressors={"regressor_cols": ["SP500", "BOND", "crisis"]}, 
         events = dict() ),
 )

import warnings

# Ignore all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
# Creates forecasts
forecaster = Forecaster()
result = forecaster.run_forecast_config(df=df, config=config)


backtest = result.backtest     # Backtest with metrics, diagnostics
result.grid_search  # Time series CV result
result.model        # Trained model
result.timeseries   # Processed time series with plotting functions
# Accesses results
forecast = result.forecast     # Forecast with metrics, diagnostics
forecast.plot()

print("Training: ")
print("MAPE: {0}".format( forecast.train_evaluation["MAPE"]))
print("RMSE: {0}".format(forecast.train_evaluation["RMSE"]))

backtest = result.backtest
print(backtest.test_evaluation) 

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Identify the columns containing the y values (exclude the index)
y_columns = df.columns

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed

# Iterate through the y value columns and plot them
for column in y_columns:
    if column == "Date":
        continue
    plt.plot(df["Date"], df[column], label=column)

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Time Series Plot of Multiple Y Values')
plt.legend()  # Show the legend to identify each line
plt.grid(True)  # Add a grid for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()

result.model[-1].summary(max_colwidth=30)

forecast.plot_components()

from greykite.framework.utils.result_summary import summarize_grid_search_results
grid_search = result.grid_search
cv_results = summarize_grid_search_results(
    grid_search=grid_search,
    decimals=2,
    # The below saves space in the printed output. Remove to show all available metrics and columns.
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
# Transposes to save space in the printed output
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
cv_results.transpose()

# backtest = result.backtest
# backtest.plot()

forecasted_mask = (forecast.df["ts"] > pd.to_datetime("2025-03-26"))
fore_df = forecast.df.loc[forecasted_mask, ]
fore_df = fore_df.dropna()
fore_df

import numpy as np
fore_mape = np.mean(abs(fore_df["actual"] - fore_df["forecast"])/(abs(fore_df["actual"]))) * 100
fore_pm = sum((fore_df["actual"] - fore_df["forecast"])**2)/sum((fore_df["actual"]-np.mean(fore_df["actual"]))**2)

print("Forecast: ")
print("MAPE: {0}".format(fore_mape))
print("PM: {0}".format(fore_pm))
