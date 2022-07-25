import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

title = "Conclusion"
sidebar_name = "Conclusion"

def run():

    st.title(title)

    st.markdown(    
        """
        <style>
        .aligncenter { text-align: center;}
       
        </style> 
        
        <h3><b>&bull; A la recherche de la features 'magique'</b></h3>
        <h3>&bull; Nécessité de cadrer d'avantage le projet</b></h3>
        <h3><b>&bull; Faibles scores de départ déstabilisant</b></h3>
        <h3><b>&bull; La quantité ne fait pas la qualité</b></h3>
        
        <p class="aligncenter"><img src='https://c.tenor.com/kQb5z-x4qpkAAAAC/game-over-insert-coins.gif' width='500'/></p>
        """, unsafe_allow_html=True)