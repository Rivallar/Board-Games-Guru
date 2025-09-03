import argparse

import chromadb
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import prompts
import settings

query_llm = Ollama(model=settings.LLM_MODEL, **settings.LLM_KWARGS)
embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)


def get_collection_files(chroma_collection) -> list[str]:
    """Fetches special chroma_collection metadata keys corresponding to names of ingested files"""
    files = []
    for k, v in chroma_collection.metadata.items():
        if v == "source file":
            files.append(k)
    return files


def get_chat_engine(collection_name: str):
    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    collection_files = get_collection_files(chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    chat_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
    retriever = VectorIndexRetriever(
        index=chat_index,
        similarity_top_k=10
        )
    # reranker = SentenceTransformerRerank(
    #     model="cross-encoder/ms-marco-MiniLM-L-2-v2,
    #     top_n=10"
    # )
    response_synthesizer = get_response_synthesizer(llm=query_llm)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)]
        )

    prompt_tmpl = PromptTemplate(
        template=prompts.QUERY_RESPONSE_TEMPLATE,
        template_var_mappings={
            "query_str": "question",
            "context_str": "context"
        }
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": prompt_tmpl}
    )
    return query_engine, collection_files


def main():
    parser = argparse.ArgumentParser(description='Pick correct db collection to chat.')
    parser.add_argument('collection_name', help='A name of a collection in ChromaDB to chat with.')
    args = parser.parse_args()

    chat_engine = get_chat_engine(collection_name=args.collection_name)
    while True:
        question = input("Enter your question: \n")
        if question == "exit":
            break
        response = chat_engine.query(question)
        print(response)


if __name__ == "__main__":
    main()
