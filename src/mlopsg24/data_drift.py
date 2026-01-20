import pandas as pd
import polars as pl
import torch
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DatasetStats, DataSummaryPreset
from loguru import logger


def monitor_drift_training_jobopslag() -> None:
    """
    What function does
    -----
    This is a demonstration of monitoring data drift in categorical data

    The functiuon compares categorical data from year 2022 to 2024.

    This mimics a data drift monotoring, where the 2022 data is used as
    reference data and 2024 data is used as the data being monitored for
    data drift.

    The function outputs 3 reports /reports/monitoring/
    """

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

    logger.info("generating report summary_training_jobopslag.html")
    (
        Report([DataSummaryPreset()])
        .run(reference_data=df_2022, current_data=df_2024)
        .save_html("reports/monitoring/summary_training_jobopslag.html")
    )

    logger.info("generating report stats_training_jobopslag.html")
    (
        Report([DatasetStats()])
        .run(reference_data=df_2022, current_data=df_2024)
        .save_html("reports/monitoring/stats_training_jobopslag.html")
    )

    logger.info("generating report drift_training_jobopslag.html")
    (
        Report([DataDriftPreset(drift_share=0.1)])
        .run(reference_data=df_2022, current_data=df_2024)
        .save_html("reports/monitoring/drift_training_jobopslag.html")
    )


def monitor_drift_embeddings() -> None:
    """
    What function does
    -----
    This is a demonstration of monitoring data drift in text embeddings.

    The function compares the embeddings in the test data to the validation.

    This mimics a data drift monotoring, where the test data is used as
    reference data and validation data is used as the data being monitored for
    data drift.

    How and why
    -----
    1. Convert the .pt files to pandas
    2. Takes a subset for demonstration - otherwise it is too slow.
    3. Use one df columns for schema - it would be an error if the columns are not identical
    4. Saves a html report in /reports/monitoring/
    """
    ts_x_test = torch.load('data/processed/x_test.pt').detach().cpu().numpy()
    df_x_test = pd.DataFrame(data=ts_x_test, columns=[f"col_{i}" for i in range(ts_x_test.shape[1])])
    df_x_test_subset = df_x_test.iloc[:1000,:100]

    ts_x_val = torch.load('data/processed/x_val.pt').detach().cpu().numpy()
    df_x_val = pd.DataFrame(data=ts_x_val, columns=[f"col_{i}" for i in range(ts_x_val.shape[1])])
    df_x_val_subset = df_x_val.iloc[:1000,:100]

    schema = DataDefinition(numerical_columns=list(df_x_val_subset.columns))

    logger.info("generating report drift_training_jobopslag.html")
    (
        Report([DataDriftPreset(embeddings=list(df_x_val_subset.columns))])
        .run(
            reference_data = (
                Dataset.from_pandas(
                    data = df_x_test_subset,
                    data_definition = schema,
                )),
            current_data=(
                Dataset.from_pandas(
                    data = df_x_val_subset,
                    data_definition = schema,
                )),
        )
        .save_html("reports/monitoring/drift_embeddings.html")
    )


if __name__ == "__main__":

    monitor_drift_training_jobopslag()
    monitor_drift_embeddings()

# #%%
# import polars as pl
# from evidently import DataDefinition, Dataset, Report
# from evidently.presets import DataDriftPreset, DataSummaryPreset, DatasetStats

# #%%
# df_raw = pl.read_parquet("data/raw/training_jobopslag.parquet")

# columns_to_inspect = ['label','erhvervsgruppe_txt','erhvervsomraade_txt',]

# schema = DataDefinition(categorical_columns=columns_to_inspect)

# df_2022 = Dataset.from_pandas(
#     data = (
#         df_raw
#         .filter(pl.col('startdt').dt.year()==
#             2022
#         )
#         .select(columns_to_inspect)
#         .to_pandas()
#     ),
#     data_definition=schema,
# )

# df_2024 = Dataset.from_pandas(
#     data = df_raw
#         .filter(pl.col('startdt').dt.year()==
#             2024
#         )
#         .select(columns_to_inspect)
#         .to_pandas(),
#     data_definition=schema,
# )

# #%%
# Report([DataSummaryPreset()]).run(reference_data=df_2022, current_data=df_2024)
# #%%
# Report([DatasetStats()]).run(reference_data=df_2022, current_data=df_2024)
# #%%
# report_datadrift = Report([DataDriftPreset()])

# report_datadrift.run(reference_data=df_2022, current_data=df_2024)


# #%%
# import torch
# import pandas as pd

# # 1. Convert the .pt files to pandas
# # 2. Take a subset for demonstration - otherwise it is too slow.
# # 3. Use one df for schema - it would be an error if the columns are not identical
# ts_x_test = torch.load('data/processed/x_test.pt').detach().cpu().numpy()
# df_x_test = pd.DataFrame(data=ts_x_test, columns=[f"col_{i}" for i in range(ts_x_test.shape[1])])
# df_x_test_subset = df_x_test.iloc[:1000,:100]

# ts_x_val = torch.load('data/processed/x_val.pt').detach().cpu().numpy()
# df_x_val = pd.DataFrame(data=ts_x_val, columns=[f"col_{i}" for i in range(ts_x_val.shape[1])])
# df_x_val_subset = df_x_val.iloc[:1000,:100]

# schema = DataDefinition(numerical_columns=list(df_x_val_subset.columns))

# #%%
# report_datadrift.run(
#     reference_data=(
#         Dataset.from_pandas(
#             data = df_x_val_subset,
#             data_definition = schema,
#         )),
#     current_data = (
#         Dataset.from_pandas(
#             data = df_x_test_subset,
#             data_definition = schema,
#         )),
# )



# # %%
