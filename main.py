import streamlit as st

from chat_engine import get_chat_engine
import utils


def go_to_step2(game_name: str):
    st.session_state.game = game_name
    st.session_state.step = 2


if "step" not in st.session_state:
    st.session_state.step = 1

st.title("Board Games Guru")
if st.session_state.step == 1:
    st.subheader("Pick a game")
    col1, col2, col3 = st.columns(3)
    game_names = utils.get_chroma_collection_names()

    for i, col in enumerate(st.columns(3)):
        this_column_games = game_names[i::3]
        with col:
            for game in this_column_games:
                st.button(game, on_click=go_to_step2, args=(game,), width="stretch")

if st.session_state.step == 2:
    chat_engine, files = get_chat_engine(st.session_state.game)
    st.subheader(st.session_state.game)

    question = st.text_area("Enter your question:")
    specific_files = st.pills("Choose specific files (optionally): ", files, selection_mode="multi")
    if st.button("Ask guru") and question:
        if specific_files:
            question += f"\nLook for the answer in this files: {', '.join(specific_files)}"
        response = chat_engine.query(question)
        st.write(utils.clear_thinking(text=response.response))
