"""
This script helps to add new documents to ChromaDB.
Just run a command in a terminal: ingestor.py path_to_document db_collection_name
"""
import argparse
import re

import chromadb
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer

import prompts
import settings

metadata_llm = Ollama(model=settings.SMALL_LLM, **settings.LLM_KWARGS)
embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)


class ThinkCleaner(TransformComponent):
    """Cleans thinking section of llm in title and questions metadata fields"""

    def __call__(self, nodes, **kwargs):
        pattern = r"<think>.*?\n</think>\n\n"
        for node in nodes:
            node.metadata["document_title"] = re.sub(pattern, "", node.metadata["document_title"])
            node.metadata["questions_this_excerpt_can_answer"] = re.sub(
                pattern, "", node.metadata["questions_this_excerpt_can_answer"]
            )
        return nodes


def get_nodes_from_a_document(doc_path: str):
    """
    Reads a given document, chunks it into pieces, and makes nodes to store in chroma db.
    Also enriches nodes with metadata: filename, title and questions tha may be answered by the node.
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL),
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    node_parser = DoclingNodeParser(chunker=chunker)
    title_extractor = TitleExtractor(nodes=5, combine_template=prompts.TITLE_GEN_TEMPLATE, llm=metadata_llm)
    qa_extractor = QuestionsAnsweredExtractor(questions=3, prompt_template=prompts.QUESTION_GEN_TMPL, llm=metadata_llm)
    pipeline = IngestionPipeline(transformations=[node_parser, title_extractor, qa_extractor, ThinkCleaner()])

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)

    docs = reader.load_data(doc_path)
    nodes = pipeline.run(documents=docs,
                         in_place=True,
                         show_progress=True
                         )

    for node in nodes:
        node.text_template = "Metadata:\n{metadata_str}\n-----\nContent:\n{content}"
        node.metadata["source_file"] = node.metadata["origin"]['filename']
        del node.metadata['doc_items']
        del node.metadata['version']
        del node.metadata['schema_name']
        del node.metadata['origin']
        if 'headings' in node.metadata:
            del node.metadata['headings']

    return nodes


def save_nodes_to_chroma(nodes, collection_name: str, file_name: str) -> None:
    """Saves nodes to chroma db."""
    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _index = VectorStoreIndex(
        nodes, storage_context=storage_context, embed_model=embed_model
    )
    chroma_collection.modify(metadata={file_name: "source file"})


def main() -> None:
    parser = argparse.ArgumentParser(description='Process a document and store it in ChromaDB')
    parser.add_argument('file_path', help='Path to the document file to process')
    parser.add_argument('collection_name', help='Name of the ChromaDB collection to store the data')
    args = parser.parse_args()

    nodes = get_nodes_from_a_document(doc_path=args.file_path)
    file_name = args.file_path.split("/")[-1]
    save_nodes_to_chroma(nodes, collection_name=args.collection_name, file_name=file_name)


if __name__ == "__main__":
    main()
