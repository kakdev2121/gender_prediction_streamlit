import streamlit as st
import utils.display as udisp
import src.gender_predict
import src.intro

MENU = {
    "Home":src.intro,
    "Specify gender" : src.gender_predict,
}

def main():
    st.sidebar.title("Menu")
    menu_selection = st.sidebar.radio(""
                                , list(MENU.keys())
    )

    menu = MENU[menu_selection]

    with st.spinner(f"Loading {menu_selection} ..."):
        udisp.render_page(menu)




if __name__ == "__main__":
    main()