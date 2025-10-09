"""
This script enriches metadata of stored nodes with node summary and questions that can be answered by the node.
"""
import argparse
import json
import logging
import sys

import chromadb
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, TransformComponent

from ingestor import MetadataCleaner
import prompts
import settings


root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
logger = logging.getLogger("board_games_guru.metadata_worker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.propagate = False


metadata_llm = Ollama(model=settings.SMALL_LLM, **settings.LLM_KWARGS)


class MetaDuplicator(TransformComponent):
    """Copies questions and section_summary metadata from top metadata level to
    _node_content as it is used in query_engine"""
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node_content = json.loads(node.metadata.get("_node_content", "{}"))
            new_data = {
                "questions_this_excerpt_can_answer": node.metadata.get("questions_this_excerpt_can_answer"),
                "section_summary": node.metadata.get("section_summary")
            }
            node_content["metadata"].update(new_data)
            node.metadata["_node_content"] = json.dumps(node_content)  
        return nodes


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
        MetadataCleaner(),
        MetaDuplicator()
        ]
    )
    processed_nodes = pipeline.run(nodes=li_nodes,
                        in_place=True,
                        show_progress=True
                        )
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
    logging.info(f"Updated {len(node_ids)} nodes with enriched metadata")


def main():
    """Enrich nodes metadata of a given collection with summary and questions that can be answered"""
    parser = argparse.ArgumentParser()
    parser.add_argument('collection_name', help='Name of the ChromaDB collection to work with')
    args = parser.parse_args()

    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    try:
        chroma_collection = db.get_collection(args.collection_name)
    except chromadb.errors.NotFoundError as e:
        logger.error(f"Wrong collection name. Exiting.\n {e}")
        sys.exit(1)
    try:
        logger.info("Metadata enrichment process has been started.")
        chroma_nodes = get_docs_from_chroma(chroma_collection=chroma_collection)
        while chroma_nodes.get("documents"):
            documents, metadatas, ids = chroma_nodes["documents"], chroma_nodes["metadatas"], chroma_nodes["ids"]
            li_nodes = create_nodes_from_chroma_data(documents=documents, metadatas=metadatas, ids=ids)
            logging.info(f"Enriching {len(li_nodes)} nodes")
            enriched_nodes = enrich_nodes_metadata(li_nodes=li_nodes)
            update_chroma_with_processed_nodes(chroma_collection=chroma_collection,
                                            enriched_nodes=enriched_nodes,
                                            node_ids=ids)

            chroma_nodes = get_docs_from_chroma(chroma_collection=chroma_collection)
        logger.info(f"{chroma_collection.name} collection is ready. Exiting.")
    except KeyboardInterrupt:
        logger.info("User stopped a process. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()