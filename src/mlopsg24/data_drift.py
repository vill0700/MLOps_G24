#%%
import polars as pl
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, DatasetStats

#%%
df_raw = pl.read_parquet("data/raw/training_jobopslag.parquet")

columns_to_inspect = ['label','erhvervsgruppe_txt','erhvervsomraade_txt',]

schema = DataDefinition(categorical_columns=columns_to_inspect)

df_2022 = Dataset.from_pandas(
    data = (
        df_raw
        .filter(pl.col('startdt').dt.year()==
            2022
        )
        .select(columns_to_inspect)
        .to_pandas()
    ),
    data_definition=schema,
)

df_2024 = Dataset.from_pandas(
    data = df_raw
        .filter(pl.col('startdt').dt.year()==
            2024
        )
        .select(columns_to_inspect)
        .to_pandas(),
    data_definition=schema,
)

#%%
Report([DataSummaryPreset()]).run(reference_data=df_2022, current_data=df_2024)
#%%
Report([DatasetStats()]).run(reference_data=df_2022, current_data=df_2024)
#%%
report_datadrift = Report([DataDriftPreset()])

report_datadrift.run(reference_data=df_2022, current_data=df_2024)


#%%
import torch
import pandas as pd

# 1. Convert the .pt files to pandas
# 2. Take a subset for demonstration - otherwise it is too slow.
# 3. Use one df for schema - it would be an error if the columns are not identical
ts_x_test = torch.load('data/processed/x_test.pt').detach().cpu().numpy()
df_x_test = pd.DataFrame(data=ts_x_test, columns=[f"col_{i}" for i in range(ts_x_test.shape[1])])
df_x_test_subset = df_x_test.iloc[:1000,:100]

ts_x_val = torch.load('data/processed/x_val.pt').detach().cpu().numpy()
df_x_val = pd.DataFrame(data=ts_x_val, columns=[f"col_{i}" for i in range(ts_x_val.shape[1])])
df_x_val_subset = df_x_val.iloc[:1000,:100]

schema = DataDefinition(numerical_columns=list(df_x_val_subset.columns))

#%%
report_datadrift.run(
    reference_data=(
        Dataset.from_pandas(
            data = df_x_val_subset,
            data_definition = schema,
        )),
    current_data = (
        Dataset.from_pandas(
            data = df_x_test_subset,
            data_definition = schema,
        )),
)



# %%
