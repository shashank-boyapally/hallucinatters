import os
import asyncio
import argparse
import faiss
import time

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext


async def main():

    # args parser
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="embedding cli for task execution"
    )
    parser.add_argument("-t", "--vector-type",   default="local", help="Type of vector db [local,faiss]")
    parser.add_argument("-c", "--chunk",   default="1024", help="Chunk size for embedding")
    parser.add_argument("-l", "--overlap",   default="10", help="Chunk overlap for embedding")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument("-m", "--model", help="LLM model used for embeddings [local, llama2, or any other supported by llama_index]")
    parser.add_argument("-o", "--output", help="Vector DB output folder")

    # execute
    args = parser.parse_args()  
    print(args)

    PERSIST_FOLDER = args.output
    INDEX_NAME = "hallucinatters_rag_index"

    # setup storage context
    match args.vector_type:
        case "local":
            print("** Local embeddings")
            storage_context = StorageContext.from_defaults()
        case "faiss":
            print("Using faiss vector store")
            faiss_index = faiss.IndexFlatL2(1536)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("** Configured storage context")

    print("Preparing node parser")
    node_parser = SimpleNodeParser.from_defaults(
        separator=" ",
        chunk_size=args.chunk,
        chunk_overlap=args.overlap,
    )

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ['OPENAI_API_KEY'] = ""
    embed_model = args.model if args.model else OpenAIEmbedding()
    Settings.llm=None
    Settings.embed_model=embed_model
    Settings.node_parser=node_parser

    print("Reading docs from local")
    documents = SimpleDirectoryReader(input_dir=args.folder, recursive=True).load_data()

    print("** Loading docs ")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    index.set_index_id(INDEX_NAME)
    index.storage_context.persist(persist_dir=PERSIST_FOLDER)
    print("*** Completed  embeddings ")

    end_time = time.time()
    execution_time_seconds = end_time - start_time
    print(f"** Total execution time in seconds: {execution_time_seconds}")

asyncio.run(main())
