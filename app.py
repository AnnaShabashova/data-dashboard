import streamlit as st
import pandas as pd

st.title("Data Dashboard")

st.write("Пример данных:")

df = pd.DataFrame({
    "Возраст": [25, 30, 35, 40],
    "Доход": [50000, 60000, 70000, 80000]
})

st.write(df)

st.bar_chart(df["Доход"])
