"""
This script helps to add new documents to ChromaDB.
Just run a command in a terminal: ingestor.py path_to_document db_collection_name
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz
import os
import tempfile
from typing import Iterable

import chromadb
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer

import settings
import utils


embed_model = HuggingFaceEmbedding(
    model_name=settings.EMBEDDING_MODEL,
    embed_batch_size=64,
    device="cpu"
    )


def _extract_page_range_to_pdf(src_path: str, start_page: int, end_page: int, out_path: str) -> None:
    """Write pages [start_page, end_page] (1-based, inclusive) from src PDF to separate out_path file."""
    src_doc = fitz.open(src_path)
    try:
        new_doc = fitz.open()
        new_doc.insert_pdf(src_doc, from_page=start_page - 1, to_page=end_page - 1)
        new_doc.save(out_path)
    finally:
        src_doc.close()


def process_document_range(doc_path: str, start_page: int, end_page: int, temp_dir: str) \
        -> tuple[Iterable[Document], int]:
    """Get documents from a separate pdf file (chunk of bigger pdf)"""
    temp_pdf_path = os.path.join(temp_dir, f"pages_{start_page}_{end_page}.pdf")
    _extract_page_range_to_pdf(doc_path, start_page, end_page, temp_pdf_path)
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    docs = reader.load_data(temp_pdf_path)
    try:
        os.remove(temp_pdf_path)
    except OSError:
        pass
    return docs, start_page


def get_docs_from_a_pdf_in_parallel(doc_path: str) -> list[Document]:
    """Process a pdf in parallel over page ranges and return list of documents."""
    print("Параллельно обрабатываем ноды")
    max_workers = int(settings.N_WORKERS)
    pdf = fitz.open(doc_path)
    try:
        num_pages = pdf.page_count
    finally:
        pdf.close()

    # Choose a reasonable range size: split into ~max_workers*2 chunks
    target_chunks = max_workers * 2
    pages_per_chunk = max(5, (num_pages + target_chunks - 1) // target_chunks)

    ranges = []
    start = 1
    while start <= num_pages:
        end = min(num_pages, start + pages_per_chunk - 1)
        ranges.append((start, end))
        start = end + 1
    print(f"Нарезали на рэнджи: {ranges}")

    with tempfile.TemporaryDirectory() as temp_dir:
        tasks = [(doc_path, s, e, temp_dir) for (s, e) in ranges]
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_document_range, *t) for t in tasks]
            for fut in as_completed(futures):
                print("Параллельный процесс что-то сделал")
                results.append(fut.result())

    results.sort(key=lambda r: r[1])
    processed_documents = []
    for doc, _start in results:
        processed_documents.extend(doc)

    return processed_documents


def get_documents_from_a_file(file_path: str) -> Iterable[Document]:
    """Reads a given file with DoclingReader and returns a list of Documents.
    Uses multiprocessing for pdf to accelerate a process for big files."""
    if file_path.endswith(".pdf"):
        docs = get_docs_from_a_pdf_in_parallel(file_path)
    else:
        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        docs = reader.load_data(file_path)
    return docs


def make_nodes(docs: Iterable[Document]) -> Iterable[BaseNode]:
    """Converts a list of DoclingReader documents into nodes."""
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
    pipeline = IngestionPipeline(transformations=[node_parser, MetadataCleaner()])

    nodes = pipeline.run(documents=list(docs), in_place=True, show_progress=False)
    return nodes


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


def save_nodes_to_chroma(nodes, collection_name: str, file_name: str) -> None:
    """Saves nodes to chroma db."""
    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    records_before = chroma_collection.count()
    print(f"Было записей в коллекции: {records_before}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _index = VectorStoreIndex(
        nodes, storage_context=storage_context, embed_model=embed_model
    )
    existing_metadata = chroma_collection.metadata or {}
    existing_metadata[file_name] = "source file"
    chroma_collection.modify(metadata=existing_metadata)
    records_now = chroma_collection.count()
    print(f"Сохранено записей в коллекции: {records_now - records_before}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Process a document and store it in ChromaDB')
    parser.add_argument('file_path', help='Path to the document file to process')
    parser.add_argument('collection_name', help='Name of the ChromaDB collection to store the data')
    args = parser.parse_args()
    documents = get_documents_from_a_file(args.file_path)
    nodes = make_nodes(documents)
    print(f"Получили {len(nodes)} нодов.")
    file_name = os.path.basename(args.file_path)
    save_nodes_to_chroma(nodes, collection_name=args.collection_name, file_name=file_name)


if __name__ == "__main__":
    main()
