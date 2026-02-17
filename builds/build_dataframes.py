import os
import numpy as np
import pandas as pd


class BuildDataframes:

    REQUIRED_COLUMNS = [
        "Time [s]",
        "Current [A]",
        "Terminal voltage [V]",
        "Cell temperature [K]",
        "Negative SEI thickness [nm]",
        "Total lithium capacity [A.h]"
    ]

    NEW_COLUMNS = [
        "Time [s]",
        "Current [A]",
        "Terminal voltage [V]",
        "Cell temperature [K]",
        "SEI Rate",
        "Lithium Capacity Rate"
    ]

    def __init__(self, data_folder):

        self.data_folder = data_folder
        self.dataframes = []

        self._load_csv_files()


    def _load_csv_files(self):

        if not os.path.isdir(self.data_folder):
            raise NotADirectoryError(
                f"{self.data_folder} is not a valid directory."
            )

        csv_files = [
            f for f in os.listdir(self.data_folder)
            if f.lower().endswith(".csv")
        ]

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_folder}"
            )

        for file_name in csv_files:

            full_path = os.path.join(self.data_folder, file_name)

            df = pd.read_csv(full_path)

            self._validate_columns(df, file_name)
            self._validate_time(df, file_name)

            df = self._rate_conversion(df)

            df = df[self.NEW_COLUMNS]

            self.dataframes.append(df)


    def _validate_columns(self, df, file_name):

        df_columns = list(df.columns)

        missing_cols = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df_columns
        ]

        if missing_cols:
            raise ValueError(
                f"File '{file_name}' is missing required columns: {missing_cols}"
            )


    def _validate_time(self, df, file_name):

        time = df["Time [s]"].values

        if not np.all(np.diff(time) > 0):
            raise ValueError(
                f"Time column is not strictly increasing in file '{file_name}'."
            )


    def _rate_conversion(self, df):

        df = df.copy()

        time = df["Time [s]"].values
        sei = df["Negative SEI thickness [nm]"].values
        li_capacity = df["Total lithium capacity [A.h]"].values

        dt = np.diff(time)
        dsei = np.diff(sei)
        dli = np.diff(li_capacity)

        rate_sei = dsei / (dt + 1e-12)
        rate_dli = dli / (dt + 1e-12)

        rate_sei = np.insert(rate_sei, 0, 0.0)
        rate_dli = np.insert(rate_dli, 0, 0.0)

        # Remove old columns
        df = df.drop(
            columns=[
                "Negative SEI thickness [nm]",
                "Total lithium capacity [A.h]"
            ]
        )

        df["SEI Rate"] = rate_sei
        df["Lithium Capacity Rate"] = rate_dli

        return df


    def get_dataframes(self):
        return self.dataframes


