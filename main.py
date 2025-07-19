import streamlit as st

#st.title("Main Page")

if st.button("Go to Image Generation"):
    st.switch_page("pages/image_generation.py")  # assumes pages/app.py

if st.button("Go to Grammatical Check"):
    st.switch_page("pages/grammatical_check.py")  # assumes pages/grammatical_check.py

if st.button("Go to Image Caption"):
    st.switch_page("pages/image_caption.py")  # assumes pages/app.py

if st.button("Go to Text summarization"):
    st.switch_page("pages/text_summarization.py")  # assumes pages/text_summarization.py

if st.button("Go to Document writer"):
    st.switch_page("pages/document_writer.py")  # assumes pages/grammatical_check.py

if st.button("Go to Research paper search"):
    st.switch_page("pages/research_paper_search.py")