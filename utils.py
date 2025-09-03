import chromadb

import settings


def clear_thinking(text: str) -> str:
    """Removes thinking block in LLM answer"""
    end = "</think>"
    end_index = text.find(end)
    cleaned_text = text[end_index + len(end):]
    return cleaned_text.strip()


def get_chroma_collection_names() -> list[str]:
    """Returns a list of chroma collection names i.e. stored games names"""
    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collections = db.list_collections()
    return [collection.name for collection in collections]