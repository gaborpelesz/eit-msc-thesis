"""
Run like:

streamlit run visualize_db.py -- --db evaluation/evaluation.db --field dataset

"""

import sqlite3
import pandas as pd
import streamlit as st

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--db", type=str, required=True)
parser.add_argument("--table", type=str, default="evaluation_results")
parser.add_argument("--field", type=str, default="dataset")

args = parser.parse_args()

# Connect to your SQLite DB
conn = sqlite3.connect(args.db)
table_name = args.table

# Load the full table into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Check if 'field' column exists
if args.field not in df.columns:
    st.error(f"'{args.field}' column not found in table '{table_name}'")
else:
    # Get unique field values
    field_values = df[args.field].dropna().unique()
    field_selected = st.selectbox("Choose a value", sorted(field_values))

    # Filter by selected dataset
    filtered_df = df[df[args.field] == field_selected]

    st.subheader(f"Rows for {args.field}: {field_selected}")
    st.dataframe(filtered_df)

    # Optional: display basic statistics if numeric columns exist
    numeric_cols = filtered_df.select_dtypes(include="number")
    if not numeric_cols.empty:
        st.subheader("Basic Summary (Numeric Columns)")
        st.write(numeric_cols.describe())
        st.bar_chart(numeric_cols)
