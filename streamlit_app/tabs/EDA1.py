import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


title = "Exploration de données 1"
sidebar_name = "Exploration de données n°1"

@st.cache #(allow_output_mutation=True)
def load_data_eda1(): 
    df = pd.read_csv('../data/vgsales.csv')
    return df

def run():

    st.title(title)
    df = load_data_eda1()
    
    st.markdown("### 1- Affichage du Dataset VGSales.csv")
    st.dataframe(df.head())
    st.markdown(f"""
    > La taille du jeux de données est de __{df.shape}__
    >
    > les chaque ligne représente un couple jeu/plateforme.""")

#
    st.markdown("### 2- Données manquantes, distribution et valeurs abberantes")
    df_col = df.isna().sum().sort_values(ascending=False).to_frame('NaN sum')
    df_col['NaN %'] =  (100*df.isna().sum()/df.shape[0]).sort_values(ascending=False)
    st.markdown("> Peu de données manquantes, uniquement Year et Publisher")
    st.dataframe(df_col.head())

#
    st.markdown("> La pluspart des ventes sont en dessous de 1.21M$, une valeur abberente en 2020 (puisque scrapé en 2017)")
    st.dataframe(df.describe(percentiles=[0.25,0.5,0.75,0.9]))

#    
    st.markdown("> Seulement trois variables explicatives categorielles")
    df_cat = df.select_dtypes(include='object')
    df_cat_modalities = pd.DataFrame({'Modalités': [df_cat['Platform'].unique().shape[0],
                                                   df_cat['Genre'].unique().shape[0],
                                                   df_cat['Publisher'].unique().shape[0]],
                                      'Description': [','.join(df_cat['Genre'].unique()),
                                                      ','.join(df_cat['Platform'].unique()),
                                                      '<too many>']},
                                index=df_cat.columns[1:])
    st.dataframe(df_cat_modalities)
    
#
    st.markdown("> Le distribution des jeux sorties par années est:")
    fig = px.histogram(df, x="Year")
    st.plotly_chart(fig, use_container_width=True, )

#
    st.markdown("> La distribution par chiffres d'affaires... sachant que le maximum en de plus de 80M$.")
    fig = px.histogram(df[df['Global_Sales']<3], x="Global_Sales")
    st.plotly_chart(fig, use_container_width=True, )
    
    #arr = df['Year']
    #fig, ax = plt.subplots(figsize=(15,5))
    #ax.hist(arr, bins=len(df['Year'].unique()))
    #st.pyplot(fig)
    
#
    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=("Top10 Genre", "Top10 revenu / genre",
                                        "Top10 Plateforme", "Top10 revenu / plateforme",
                                        "Top10 Publisher", "Top10 revenu / éditeur"))

    gg = df.groupby("Genre",as_index=False).agg({'Global_Sales': 'sum'}).sort_values('Global_Sales',ascending=False)
    pp = df.groupby("Platform",as_index=False).agg({'Global_Sales': 'sum'}).sort_values('Global_Sales',ascending=False)
    p2 = df.groupby("Publisher",as_index=False).agg({'Global_Sales': 'sum'}).sort_values('Global_Sales',ascending=False)

    fig.add_trace(
        go.Bar(
        x=df["Genre"].value_counts()[:10].index,
        y=df["Genre"].value_counts()[:10],
        name="Genre"),
        row=1, col=1)
    fig.add_trace(
        go.Bar(
        x=df["Platform"].value_counts()[:10].index,
        y=df["Platform"].value_counts()[:10],
        name="Platform"),
        row=2, col=1)
    fig.add_trace(
        go.Bar(
        x=df["Publisher"].value_counts()[:10].index,
        y=df["Publisher"].value_counts()[:10],
        name="Publisher"),
        row=3, col=1)
    fig.add_trace(
        go.Bar(
        x=gg['Genre'].values[:10],
        y=gg['Global_Sales'].values[:10],
        name="M$ Genre"),
        row=1, col=2)
    fig.add_trace(
        go.Bar(
        x=pp['Platform'].values[:10],
        y=pp['Global_Sales'].values[:10],
        name="M$ Platform"),
        row=2, col=2)
    fig.add_trace(
        go.Bar(
        x=p2['Publisher'].values[:10],
        y=p2['Global_Sales'].values[:10],
        name="M$ Publisher"),
        row=3, col=2)
    fig.update_layout(height=800, width=1000, title_text="Top 10 par quantité (à gauche) et par revenu (à droite)")
    st.plotly_chart(fig, use_container_width=True, )
    st.markdown("> Ce n'est pas ceux qui se vendent le plus qui rapportent le plus !")
    
    
