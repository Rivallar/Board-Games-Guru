"""
This script helps to add new documents to ChromaDB.
Just run a command in a terminal: ingestor.py path_to_document db_collection_name
"""
import argparse
import os

import chromadb
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from llama_index.core.extractors import (
    SummaryExtractor,
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
import utils

metadata_llm = Ollama(model=settings.SMALL_LLM, **settings.LLM_KWARGS)
embed_model = HuggingFaceEmbedding(
    model_name=settings.EMBEDDING_MODEL,
    embed_batch_size=64,
    device="cpu"
    )


class MetadataCleaner(TransformComponent):
    """Transforms metadata dictionary to a desired state"""

    @classmethod
    def clean_thinking(cls, text: str) -> str:
        """Cleans thinking section of llm in metadata fields"""
        if not isinstance(text, str):
            return ""
        end_pattern = "</think>"
        end_ind = text.find(end_pattern)
        if end_ind == -1:
            return text

        cleaned_text = text[end_ind + len(end_pattern):].strip()
        return cleaned_text

    def __call__(self, nodes, **kwargs):

        for node in nodes:
            node.text_template = "Metadata:\n{metadata_str}\n-----\nContent:\n{content}"
            origin = node.metadata.get("origin", {})
            node.metadata["source_file"] = origin.get("filename", node.metadata.get("file_name", "unknown"))
            if "questions_this_excerpt_can_answer" in node.metadata:
                node.metadata["questions_this_excerpt_can_answer"] = self.clean_thinking(
                    node.metadata["questions_this_excerpt_can_answer"])
            for key in getattr(node, "excluded_embed_metadata_keys", []):
                node.metadata.pop(key, None)
            if "headings" in node.metadata:
                node.metadata["headings"] = ", ".join(node.metadata["headings"])
            for summary in ("prev_section_summary", "next_section_summary", "section_summary"):
                if summary in node.metadata:
                    node.metadata[summary] = self.clean_thinking(node.metadata[summary])
            node.id_ = utils.make_stable_node_id(node.get_content(), node.metadata["source_file"])
        return nodes


def get_nodes_from_a_document(doc_path: str):
    """
    Reads a given document, chunks it into pieces, and makes nodes to store in chroma db.
    Also enriches nodes with metadata: filename, title and questions tha may be answered by the node.
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL),
        max_tokens=settings.MAX_CHUNK_TOKENS
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        overlap_tokens=settings.CHUNK_OVERLAP
    )

    node_parser = DoclingNodeParser(chunker=chunker)
    summary_extractor = SummaryExtractor(
        summaries=["self"],
        llm=metadata_llm,
        prompt_template=prompts.SUMMARY_TMPL,
    )
    qa_extractor = QuestionsAnsweredExtractor(questions=1, prompt_template=prompts.QUESTION_GEN_TMPL, llm=metadata_llm)
    pipeline = IngestionPipeline(transformations=[node_parser,
        # summary_extractor,
        # qa_extractor, 
        MetadataCleaner()]
        )

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)

    docs = reader.load_data(doc_path)
    nodes = pipeline.run(documents=docs,
                         in_place=True,
                         show_progress=True
                         )
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
    existing_metadata = chroma_collection.metadata or {}
    existing_metadata[file_name] = "source file"
    chroma_collection.modify(metadata=existing_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description='Process a document and store it in ChromaDB')
    parser.add_argument('file_path', help='Path to the document file to process')
    parser.add_argument('collection_name', help='Name of the ChromaDB collection to store the data')
    args = parser.parse_args()

    nodes = get_nodes_from_a_document(doc_path=args.file_path)
    file_name = os.path.basename(args.file_path)
    save_nodes_to_chroma(nodes, collection_name=args.collection_name, file_name=file_name)


if __name__ == "__main__":
    main()
