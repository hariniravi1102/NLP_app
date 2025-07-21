import streamlit as st

import streamlit as st

st.set_page_config(
    page_title="AI TOOLS",
    layout="wide",
    initial_sidebar_state="collapsed"
)
#st.title("Main Page")
# Completely hide the sidebar toggle button
hide_sidebar = """
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

#if st.button("Go to Article_writer"):
#   st.switch_page("pages/document_writer.py")

if st.button("Go to Grammatical Check"):
    st.switch_page("pages/grammatical_check.py")

if st.button("Go to Text summarization"):
    st.switch_page("pages/text_summarization.py")

if st.button("Go to Research paper search"):
    st.switch_page("pages/research_paper_search.py")

if st.button("Go to chat"):
    st.switch_page("pages/question_and_answering.py")

if st.button("Go to Image Generation"):
    st.switch_page("pages/image_generation.py")

if st.button("Go to Image Caption"):
    st.switch_page("pages/image_caption.py")