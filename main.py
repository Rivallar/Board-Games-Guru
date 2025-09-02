import streamlit as st

from chat_engine import get_chat_engine


def clear_thinking(text: str) -> str:
    """Removes thinking block in LLM answer"""
    end = "</think>"
    end_index = text.find(end)
    cleaned_text = text[end_index + len(end):]
    return cleaned_text.strip()


st.title("Board Games Guru")
chat_engine = get_chat_engine("108_fool")

question = st.text_area("Enter your question.")
if st.button("Ask guru") and question:
    response = chat_engine.query(question)
    st.write(clear_thinking(text=response.response))
