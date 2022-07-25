import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
title = "Exploration de données "
sidebar_name = "Exploration de données "

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
    > La taille du jeu de données est de __{df.shape}__
    >
    > Chaque ligne représente un couple jeu/plateforme.""")

#
    st.markdown("### 2- Données manquantes, distribution et valeurs aberantes")
    df_col = df.isna().sum().sort_values(ascending=False).to_frame('NaN sum')
    df_col['NaN %'] =  (100*df.isna().sum()/df.shape[0]).sort_values(ascending=False)
    st.markdown("> Peu de données manquantes, uniquement Year et Publisher")
    st.dataframe(df_col.head())

#
    st.markdown("> La plupart des ventes sont en dessous de 1.21M$, une valeur aberante en 2020 (puisque scrapé en 2017)")
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
    st.markdown("> La distribution des jeux sortis par année est:")
    fig = px.histogram(df, x="Year")
    st.plotly_chart(fig, use_container_width=True, )

#
    st.markdown("> La distribution par chiffres d'affaires... sachant que le maximum est de plus de 80M$.")
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

    st.markdown("### 3- Exploration des données après scraping")
    wiki = pd.read_csv('../data/wiki_merge.csv', sep  = ';')
    st.write('Nombre de lignes après le merge')
    shapes =[df.shape[0], wiki.shape[0]]
    st.plotly_chart(px.bar(x = ['vgsales','wiki_merged_data'],y = shapes),use_container_width=True,)
 
    df1 = wiki.groupby('series')['Global_Sales'].sum().reset_index()
    df2 = df1.sort_values(by = 'Global_Sales',ascending = False)
    df3 = df2[df2.series != 'NO FRANCHISE']
    df4 = wiki.drop_duplicates(subset=['Name'])

    df5 = df4.groupby('series')['Name'].count().reset_index()

    df5 = df5.sort_values(by = 'Name',ascending = False)

    df5 = df5.rename(columns={'Name':'count_games'})

    df5 = df5[df2.series != 'NO FRANCHISE']


    
   
    fig2 = make_subplots(rows=1, cols=2,
                        subplot_titles=("Top vente par franchise",
                                        "Top nombre de jeux par franchise"))

   
    fig2.add_trace(
        go.Bar(
        x = df3['series'].values[:10],
        y = df3['Global_Sales'].values[:10],
        name="Franchise"),
        row=1, col=1)
    fig2.add_trace(
        go.Bar(
        x = df5['series'].values[:10],
        y = df5['count_games'].values[:10],
        name="Franchise"),
        row=1, col=2)
    fig2.update_layout(height=800, width=1000)
    st.plotly_chart(fig2, use_container_width=True, )

    st.write('Les franchises qui vendent le plus ne sont pas forcément celles qui produisent plus de versions.')

    st.markdown('### Corrélation vente / nombre de jeux par franchise / nombre de jeux primés par franchise')
    #ajout de la variable awards dans un sunburst
    df6 = df3.merge(df5,how = 'inner', on  = 'series')
    df6.iloc[5,1] = 109.46
    
    df7 = df4.groupby('series')['awarded'].sum().reset_index()

    df8 = df7.sort_values(by = 'awarded',ascending = False)
    df9 =  df6.merge(df8,how = 'inner', on  = 'series')

    st.dataframe(df9.corr())
    st.write('- On constate une corrélation entre 0.5 et 0.7 donc une corrélation modérée \n')
    st.write('- 8 jeux parmi 20 franchises les plus vendues sont dans le top 20 des franchises avec le plus grand nombre de jeux')
    st.write('- Les prix sont plus corrélés avec les ventes que le nombre de jeux produits')
    st.write('- La franchise GTA qui a produit 9 jeux seulement est la 4 éme franchise la plus vendue comme le montre le sunburst ci-dessous. Le fait que cette franchise ait 7 jeux primés parmis 9 explique')
    st.write('son succès')

    fig3 = px.sunburst(df9.head(10),
                  path=['series', 'Global_Sales','count_games','awarded'],
                  values = 'Global_Sales',
                  title="Sunburst global sales , nombre de jeux & total des jeux primés par franchise ",
                  width=750, height=750)
    st.plotly_chart(fig3, use_container_width=True, )

    st.markdown('# Analyse de la variable Mode de jeu : Multiplayer / SinglePlayer')

    game_mode = pd.read_csv('../data/vgnew9.csv')
    game_mode = game_mode.drop(['Unnamed: 0','Rank'],axis=1)
    game_mode = game_mode.dropna()
    st.write('- Les jeux multijoueurs sont plus présents que les jeux singleplayer')
    st.write(game_mode.multiplayer.value_counts(normalize = True))
    game_mode.Year = game_mode.Year.astype('int64')

    st.write("### Evolution des types de jeux ( multiplayer / single player ) selon le genre de jeux et l'année")
    
    for g in game_mode.Genre.unique():
        
        sns.catplot(x='Year', kind='count', data= game_mode[game_mode['Genre']==g] , hue = 'multiplayer',  height=2, aspect=8 )
        plt.title(g)
        st.pyplot()

    st.markdown('Mode de jeu par Genre et par plateforme')
    sns.catplot(x='Genre', kind='count', data= game_mode , hue = 'multiplayer',  height=2, aspect=8 )
    st.pyplot()
    st.write('- On remarque que la plupart des jeux sont des jeux multijoueurs, sauf pour les jeux de types : Adventure,')
    st.write('Role-playing et Platform où les jeux sont plus souvent des jeux solos')
    sns.catplot(x='Platform', kind='count', data= game_mode , hue = 'multiplayer',  height=2, aspect=8 )
    st.pyplot()
    st.write('- Seules les plateformes PS Vita ,NES , SNES, GB et  SAT ont plus de jeux singleplayer que de jeux multiplayer')


    game_mode2 = game_mode.groupby(['Publisher','multiplayer']).agg({'multiplayer': ['count']}).reset_index()
    game_mode2.columns = ['Publisher','multiplayer','count']
    game_mode2 = game_mode2.sort_values(by = ['multiplayer','count'], ascending = False)

    df10 = game_mode2[game_mode2['multiplayer']== True].head(15)
    df11 = game_mode2[game_mode2['multiplayer']== False].head(15)

    fig5, [ax1, ax2] = plt.subplots(nrows=1, ncols=2,figsize=(10,5))

    ax1.bar(x = 'Publisher', height = 'count', data = df10,color = 'r')
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_title('game mode multi  by publisher ')



    ax2.bar(x = 'Publisher', height = 'count', data = df11)
    ax2.tick_params(axis='x', labelrotation=90)
    ax2.set_title('game mode single by publisher ')
    st.pyplot(fig5)

    st.markdown('# Analyse relation Global Sales / nombre de vues Youtube')

    st.write('- le Dataset scrapé de Youtube est le suivant')
    yt = pd.read_csv('../data/vgnew_nodesc3.csv',sep = ';')
    yt = yt.dropna()
    st.dataframe(yt.head(5))
    fig6= plt.figure(figsize= (10,5))
    st.write('- Relation vente, nombre de likes sur les reviews Youtube')
    sns.lineplot(data=yt, y='Global_Sales', x='likecount')
    
    st.pyplot(fig6)
    st.write('- On ne peut pas déterminer une vraie relation entre le nombre de vues / nombre de likes sur Youtube et les ventes des jeux , le dataset Youtube ne sera de ce fait pas utilisé dans notre analyse.')

    st.markdown('# Analyse relation Global Sales / notes metacritics')
    
    meta = pd.read_csv('../data/metacritic.csv')
    meta_merge = wiki.merge(meta,how = 'inner', on = 'Name')
    meta_merge = meta_merge.drop(['Platform_x','Year','Genre_x','Publisher_x','Platform_y', 'Release_date',
       'Genre_y','Publisher_y','Developer'],axis = 1)
    functions = {
    'Meta_Critic_Score':np.mean,
    'Meta_User_Score':np.mean,
    'N_Critic_Reviews' : np.mean,
    'N_User_Reviews' : np.mean ,
    'Global_Sales' : np.sum
    }
    df12 = meta_merge.groupby(['Name','multiplayer', 'singleplayer', 'awarded', 'series']).agg(functions).reset_index()
    st.write("- Ce dataset est une fusion du dataset scrapé de Wikipedia et celui qu'on a scrapé de Metacritics")
    st.dataframe(df12[df12['series'] != 'NO FRANCHISE'].head())

    st.write('- Nombre de lignes après le merge')
    shapes =[df.shape[0], wiki.shape[0],df12.shape[0]]
    st.plotly_chart(px.bar(x = ['vgsales','wiki_merged_data','metacritics_merged_data'],y = shapes),use_container_width=True,)

    st.write('- Matrice de corrélation du nouveau dataset')
    cor = df12.corr()

    fig7, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(cor, annot=True, ax=ax, cmap='coolwarm')
    st.pyplot(fig7)

    st.write("- On remarque que le mode de jeu n'est pas corrélé avec les notes")
    st.write("- Les notes, le nombre de users qui ont noté ainsi que la variable awards sont légérement corrélés positivement")

    fig8 = plt.figure(figsize  = (10,5))
    sns.lineplot(df12.Meta_User_Score,df12.Global_Sales)
    plt.title('Ventes en fonction des notes utilisateurs')
    st.pyplot(fig8)

    





