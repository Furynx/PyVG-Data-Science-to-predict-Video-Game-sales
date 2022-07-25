import streamlit as st


title = "Data Science to predict Video Games Sales"
sidebar_name = "Introduction"

def run():
    st.markdown(
        """
        <style>
        .aligncenter { text-align: center;}
       
        </style>  
        <p class="aligncenter">
        <img src='https://assets.morningconsult.com/wp-uploads/2021/11/04170352/211104_Gaming-Series_Story-3_Subscription-Services_FI.gif' width=800 align=center />
        </p>
        """
        ,unsafe_allow_html=True)

    st.title(title)
    #st.markdown("---")
    st.image("assets/intro.png", width=900)

    # Choose between one of these GIFs 1 to 3
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    #
    st.markdown("")
    st.markdown(
        """
        #### Introduction
        #### 1- Présentation et constitution du dataset (Stéphane)
        #### 2- Analyse exploratoire (Hsan)
        #### 3- Modélisation - Régression et Classification (Henri-François)
        #### 4- Mise en application (Alexis)
        #### Conclusion
        """
    )
