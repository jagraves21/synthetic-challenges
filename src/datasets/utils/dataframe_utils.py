import pandas as pd
from IPython.display import display

def describe_df(df, include="all"):
	description_df = df.describe(include=include).T

	description_df = description_df.reindex(df.columns)

	description_df["non_null"] = df.notnull().sum().astype("Int64")
	description_df["null_count"] = df.isnull().sum().astype("Int64")
	description_df["dtype"] = df.dtypes

	if "count" in description_df.columns:
		description_df["count"] = description_df["count"].astype("Int64")

	base_cols = ["dtype", "count", "non_null", "null_count"]
	stat_cols = [col for col in description_df.columns if col not in base_cols]
	ordered_cols = base_cols + stat_cols

	return description_df[ordered_cols]

def print_dataframe_info(df, name="DataFrame"):
	mem_bytes = df.memory_usage(deep=True).sum()
	mem_megabytes = mem_bytes / 1024**2
	print(f"{name} Memory Usage: {mem_megabytes:.2f} MB")

	if df.empty or df.columns.size == 0:
		print(f"{name} is empty or has no columns.")
	else:
		with pd.option_context("display.max_rows", None, "display.max_columns", None):
			display(describe_df(df))

		with pd.option_context("display.max_rows", 10, "display.max_columns", None):
			display(df)

