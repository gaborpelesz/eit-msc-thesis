"""
Run like:

streamlit run visualize_db.py -- --db evaluation/evaluation.db --fields dataset tolerance

"""

import sqlite3
import pandas as pd
import streamlit as st

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--db", type=str, required=True)
parser.add_argument("--table", type=str, default="evaluation_results")
parser.add_argument(
    "--fields",
    type=str,
    nargs="+",
    default=["dataset"],
    help="List of fields to filter by (e.g., --fields dataset tolerance)",
)

args = parser.parse_args()

# Connect to your SQLite DB
conn = sqlite3.connect(args.db)
table_name = args.table

# Load the full table into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Check if all specified fields exist
missing_fields = [field for field in args.fields if field not in df.columns]
if missing_fields:
    st.error(
        f"Column(s) {missing_fields} not found in table '{table_name}'"
    )
else:
    selected_values = {}
    for field in args.fields:
        field_values = df[field].dropna().unique()
        # Use selectbox for each field
        selected = st.selectbox(
            f"Choose a value for '{field}'", sorted(field_values), key=field
        )
        selected_values[field] = selected

    # Filter DataFrame by all selected field values
    filtered_df = df.copy()
    for field, value in selected_values.items():
        filtered_df = filtered_df[filtered_df[field] == value]

    st.subheader(
        "Rows for: "
        + ", ".join(
            f"{field}: {selected_values[field]}" for field in args.fields
        )
    )
    st.dataframe(filtered_df)

    # Optional: display basic statistics if numeric columns exist
    numeric_cols = filtered_df.select_dtypes(include="number")
    if not numeric_cols.empty:
        st.subheader("Basic Summary (Numeric Columns)")
        st.write(numeric_cols.describe())
        st.bar_chart(numeric_cols)
