import streamlit as st
import utils.display as udisp

def write():
    udisp.render_md("resources/home_intro.md")
    st.image('src/intro.png', caption='', width=600, use_column_width=False, clamp=False, channels='RGB')