import streamlit as st
from numpy import median
import pandas as pd

st.title("Friend Circle Size Estimator")
st.markdown(
    """ 
    This app allows for educators to estimate the number a friends a student may have based on
    self-reported time spent alone, posting online, going outside, attending events, and
    attending outside events.
    """
)

pre_df = pd.read_csv("data/preprocessing_dataframe.csv")
predictors = ["Time_spent_Alone", "Post_frequency", "Going_outside", "Social_event_attendance"]
column_range: dict[str, dict[str, int | float]] = {}
for column in predictors:
    column_range[column] = {
        "max": max(pre_df[column]),
        "median": median(pre_df[column]),
        "min": min(pre_df[column])
    }

humanized_label = {
    "Time_spent_Alone": "Hours spent alone",
    "Post_frequency": "Number of posts online",
    "Going_outside": "Number of instances going outside",
    "Social_event_attendance": "Number of social events attended"
}

result = {}
for column in predictors:
    # humanized_label: str = ' '.join(column.lower().split('_')).capitalize()
    value = st.slider(
        humanized_label[column],
        min_value=column_range[column]["min"],
        max_value=column_range[column]["max"],
        value=column_range[column]["median"],
    )
    result[column] = value

coef_df = pd.read_csv("data/coefficient_dataframe.csv", index_col=0)


def estimate(x, column):
    coefficient: float = coef_df.loc[column]
    proportion: float = x / column_range[column]["max"]
    return coefficient * proportion


def get_estimate():
    estimated = 0.0
    for column in predictors:
        estimated += estimate(result[column], column)
    return round(estimated.loc["Coefficient"], 2)


if st.button("Estimate"):
    st.text("{} friends".format(get_estimate()))
