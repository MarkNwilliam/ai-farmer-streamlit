#AIzaSyA3YVLTGuGesO27kFo1QGZq-lPNebj3ihg
import streamlit as st
from app_pages import home
import overview
import analysis
import  standards
import animalstandards
import data_analysis


def main():
    st.sidebar.title("Navigate")
    page = st.sidebar.radio("Go to", ["Home", "Overview", "Analysis","Check Animal farming Standards", "Check Crop farming Standards", "Data Analysis"])

    if st.session_state.get("current_page") != page:
        # Clear the chat session state
        st.session_state["messages"] = []

    if page == "Home":
        home.show()
    elif page == "Overview":
        overview.show()
    elif page == "Analysis":
        analysis.show()

    elif page == "Check Crop farming Standards":
        standards.show()
    elif page == "Check Animal farming Standards":
        animalstandards.show()
    elif page == "Data Analysis":
        data_analysis.show()

    # Store the current page in the session state
    st.session_state["current_page"] = page

if __name__ == "__main__":
    main()
