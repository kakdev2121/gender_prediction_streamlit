import base64
import importlib
import streamlit as st

def  render_page(menupage):
    menupage.write()

def get_file_content_as_string(path):
    response = open(path, encoding="utf-8").read()
    return response

def render_md(md_file_name):
    st.markdown(get_file_content_as_string(md_file_name))