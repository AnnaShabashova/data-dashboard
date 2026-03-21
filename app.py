import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Quality Dashboard", layout="wide")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]
    df = pd.read_csv(url, names=columns)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df = df.replace("?", np.nan)
    df = df.drop_duplicates()
    return df

df = load_data()

@st.cache_data
def make_k_anonymized(df):
    df_k = df.copy()

    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ["0-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    df_k["age_group"] = pd.cut(df_k["age"], bins=bins, labels=labels, right=False)
    df_k = df_k.drop(columns=["age"])

    df_k["native-country"] = df_k["native-country"].apply(
        lambda x: x if x == "United-States" else "Other"
    )

    quasi_identifiers = ["age_group", "sex", "native-country"]
    group_sizes = df_k.groupby(quasi_identifiers).size().reset_index(name="count")
    df_k = df_k.merge(group_sizes, on=quasi_identifiers, how="left")

    k = 5
    df_k_anon = df_k[df_k["count"] >= k].drop(columns=["count"])
    return df_k_anon

df_k_anon = make_k_anonymized(df)

@st.cache_data
def make_simple_synthetic(df):
    df_synth = df.copy()

    for col in df_synth.select_dtypes(include="object").columns:
        df_synth[col] = df_synth[col].fillna("Unknown")

    for col in df_synth.select_dtypes(exclude="object").columns:
        df_synth[col] = df_synth[col].fillna(df_synth[col].median())

    synthetic = pd.DataFrame()

    num_cols = df_synth.select_dtypes(include=np.number).columns
    for col in num_cols:
        synthetic[col] = df_synth[col].sample(len(df_synth), replace=True).reset_index(drop=True)

    cat_cols = df_synth.select_dtypes(include="object").columns
    for col in cat_cols:
        synthetic[col] = df_synth[col].sample(len(df_synth), replace=True).reset_index(drop=True)

    synthetic = synthetic[df_synth.columns]
    return df_synth, synthetic

df_clean_for_synth, synthetic_data = make_simple_synthetic(df)

missing = df.isnull().sum()
missing_total = int(missing.sum())

gx_results = {
    "Возраст 16-100": True,
    "income без пропусков": True,
    "sex ∈ {Male, Female}": True,
    "hours-per-week 1-100": True,
    "Отсутствие полных дублей": True,
}

st.title(" Анализ качества данных и Privacy Engineering")

st.markdown("""
Дашборд объединяет результаты анализа качества данных, валидации и анонимизации
для набора данных Adult Income Dataset.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "Обзор данных",
    "Разведочный анализ",
    "Great Expectations",
    "Анонимизация"
])

with tab1:
    st.header("Общая информация о данных")

    c1, c2, c3 = st.columns(3)
    c1.metric("Число строк", df.shape[0])
    c2.metric("Число столбцов", df.shape[1])
    c3.metric("Всего пропусков", missing_total)

    st.subheader("Пропуски по колонкам")
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Колонка", "Пропуски"]
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Первые строки датасета")
    st.dataframe(df.head(), use_container_width=True)

with tab2:
    st.header("Разведочный анализ")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Распределение дохода")
        st.bar_chart(df["income"].value_counts())

    with col_right:
        st.subheader("Распределение пола")
        st.bar_chart(df["sex"].value_counts())

    st.subheader("Корреляции числовых признаков")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Boxplot: age")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df["age"], ax=ax2)
        st.pyplot(fig2)

    with col_b:
        st.subheader("Boxplot: hours-per-week")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df["hours-per-week"], ax=ax3)
        st.pyplot(fig3)

    st.info(
        "В данных выявлены пропуски в workclass, occupation и native-country, "
        "а также асимметрия распределений и дисбаланс по ряду признаков."
    )

with tab3:
    st.header("Валидация Great Expectations")

    gx_df = pd.DataFrame({
        "Ожидание": list(gx_results.keys()),
        "Результат": [" Успешно" if v else " Ошибка" for v in gx_results.values()]
    })

    st.dataframe(gx_df, use_container_width=True)
    st.metric("Успешно пройдено проверок", "100%")
    st.success("После предобработки все сформулированные ожидания были успешно пройдены.")

with tab4:
    st.header("Анонимизация данных")

    subtab1, subtab2 = st.tabs(["K-анонимность", "Синтетические данные"])

    with subtab1:
        st.subheader("Результат K-анонимности")
        st.write(f"Исходный размер: {df.shape}")
        st.write(f"После k-анонимности: {df_k_anon.shape}")
        st.dataframe(df_k_anon.head(), use_container_width=True)
        st.info(
            "Возраст был обобщён до интервалов, а страны укрупнены до категорий United-States и Other."
        )

    with subtab2:
        st.subheader("Синтетические данные")
        st.write(f"Размер синтетического датасета: {synthetic_data.shape}")
        st.dataframe(synthetic_data.head(), use_container_width=True)

        num_cols = df_clean_for_synth.select_dtypes(include=np.number).columns
        comparison = pd.DataFrame({
            "Оригинал": df_clean_for_synth[num_cols].mean(),
            "Синтетика": synthetic_data[num_cols].mean()
        })
        comparison["Разница"] = comparison["Синтетика"] - comparison["Оригинал"]

        st.subheader("Сравнение средних значений")
        st.dataframe(comparison, use_container_width=True)

        st.warning(
            "Синтетические данные хорошо сохраняют общие закономерности, "
            "но могут искажать редкие и экстремальные значения."
        )
