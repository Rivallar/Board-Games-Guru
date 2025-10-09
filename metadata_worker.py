"""
This script enriches metadata of stored nodes with node summary and questions that can be answered by the node.
"""
import argparse
import json
import sys

import chromadb
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode

from ingestor import MetadataCleaner
import prompts
import settings


metadata_llm = Ollama(model=settings.SMALL_LLM, **settings.LLM_KWARGS)


def get_docs_from_chroma(chroma_collection, limit: int = 10) -> dict:
    """Fetches N nodes without summary metadata from chroma"""
  
    chroma_nodes = chroma_collection.get(limit=limit, 
                                        where={"section_summary": "empty"}, 
                                        include=["documents", "metadatas", "embeddings"])
    return chroma_nodes


def create_nodes_from_chroma_data(documents: list[str], metadatas: list[dict], ids: list[str]) -> list[TextNode]:
    """Convert ChromaDB data back to LlamaIndex nodes"""
    li_nodes = []
    for doc, metadata, node_id in zip(documents, metadatas, ids):
        node = TextNode(
            text=doc,
            metadata=metadata,
            id_=node_id
        )
        li_nodes.append(node)
    
    return li_nodes


def enrich_nodes_metadata(li_nodes: list[TextNode]) -> list[TextNode]:
    """Adds section_summary and questions_this_excerpt_can_answer fields to nodes metadata"""
    summary_extractor = SummaryExtractor(
        summaries=["self"],
        llm=metadata_llm,
        prompt_template=prompts.SUMMARY_TMPL,
    )
    qa_extractor = QuestionsAnsweredExtractor(questions=3, prompt_template=prompts.QUESTION_GEN_TMPL, llm=metadata_llm)
    pipeline = IngestionPipeline(transformations=[
        summary_extractor,
        qa_extractor,
        MetadataCleaner()
        ]
    )
    processed_nodes = pipeline.run(nodes=li_nodes,
                        in_place=True,
                        show_progress=True
                        )
    for node in processed_nodes:
    node_content = json.loads(current_metadata.get("_node_content", "{}"))    
    return processed_nodes


def update_chroma_with_processed_nodes(chroma_collection, enriched_nodes: list[TextNode], node_ids: list[str]) -> None:
    """Update ChromaDB collection nodes with new metadata"""
    
    enriched_metadatas = []
    for node in enriched_nodes:        
        enriched_metadatas.append(node.metadata)
    
    chroma_collection.update(
        ids=node_ids,
        metadatas=enriched_metadatas
    )  
    print(f"Updated {len(node_ids)} nodes with enriched metadata")


def main():
    """Enrich nodes metadata of a given collection with summary and questions that can be answered"""
    parser = argparse.ArgumentParser()
    parser.add_argument('collection_name', help='Name of the ChromaDB collection to work with')
    args = parser.parse_args()

    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    try:
        chroma_collection = db.get_collection(args.collection_name)
    except chromadb.errors.NotFoundError as e:
        print(e)
        sys.exit(1)
    try:
        chroma_nodes = get_docs_from_chroma(chroma_collection=chroma_collection)
        while chroma_nodes.get("documents"):
            documents, metadatas, ids = chroma_nodes["documents"], chroma_nodes["metadatas"], chroma_nodes["ids"]
            li_nodes = create_nodes_from_chroma_data(documents=documents, metadatas=metadatas, ids=ids)
            enriched_nodes = enrich_nodes_metadata(li_nodes=li_nodes)
            update_chroma_with_processed_nodes(chroma_collection=chroma_collection,
                                            enriched_nodes=enriched_nodes,
                                            node_ids=ids)

            chroma_nodes = get_docs_from_chroma(chroma_collection=chroma_collection)
    except KeyboardInterrupt:
        print(" Exiting")
        sys.exit(0)


if __name__ == "__main__":
    main()