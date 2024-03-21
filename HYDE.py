import os
import warnings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings, load_index_from_storage
from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from llama_index.core import PromptHelper, StorageContext, Settings,get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain_core._api.deprecation import LangChainDeprecationWarning
import requests
requests.packages.urllib3.disable_warnings()

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
class Hyde:
    def __init__(self,args):
        self.args=args
        print("Preparing node parser")
        node_parser = SimpleNodeParser.from_defaults(
        separator=" ",
        chunk_size=args.chunk,
        chunk_overlap=args.overlap,
        )

        print("Initializing prompt helper")
        prompt_helper = PromptHelper(
        context_window=4096, 
        num_output=256, 
        chunk_overlap_ratio=0.1, 
        chunk_size_limit=None
        )

        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        os.environ['OPENAI_API_KEY'] = ""
        embed_model = args.model if args.model else OpenAIEmbedding()
        Settings.llm=None
        Settings.embed_model=embed_model
        Settings.node_parser=node_parser
        Settings.prompt_helper=prompt_helper

        server_url = ""
        self.hyde_llm = HuggingFaceTextGenInference(
            inference_server_url=server_url,
            max_new_tokens=512,
            top_k=50,
            top_p=0.99,
            typical_p=0.99,
            temperature=1,
            repetition_penalty=1.03,
            do_sample=True,
        )

        self.llm = HuggingFaceTextGenInference(
            inference_server_url=server_url,
            max_new_tokens=512,
            top_k=10,
            top_p=0.5,
            typical_p=0.5,
            temperature=0.01,
            repetition_penalty=1.03,
            streaming=True,
        )

        print("Setting up storage context for index load...")
        vector_store = FaissVectorStore.from_persist_dir(args.persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=args.persist_dir
        )

        print("Loading up the index")
        index = load_index_from_storage(
            index_id=args.index_name,
            storage_context=storage_context 
        )

        print("Fetching similar docs and summarizing them")
        ret= VectorIndexRetriever(
            index=index,
            similarity_top_k=1
        )
        response_syn=get_response_synthesizer(
            response_mode="tree_summarize",
        )
        self.query_engine= RetrieverQueryEngine(
            retriever=ret,
            response_synthesizer=response_syn,
        )

    def queryHyde(self,query):
        hyde_prompt = """
        <question>
        Please generate a well-structured and informative document that answers the following question in a hypothetical but relevant manner:
        <question_text>

        The generated document should follow this outline:

        1. Introduction to the topic and its relevance
        2. Key points or sections to be covered in the document
        - Point 1
        - Point 2
        - ...
        3. Additional aspects or considerations
        4. Conclusion and future outlook

        Please provide a well-written and engaging document that covers the outlined points, with examples, insights, and meaningful content related to the question. The document should be hypothetical but still relevant and informative.
        </question>
        """

        # Replace <question_text> with your actual question
        print("Generating hypothetical document...")
        hyde_prompt = hyde_prompt.replace("<question_text>", query)
        generated_document = self.hyde_llm(hyde_prompt)

        response = self.query_engine.query(generated_document)
        file_dict=dict()
        for context in response.source_nodes:
            file_path = context.metadata.get('file_path')
            if file_path not in file_dict:
                file_dict[file_path] = context.get_text()
            else:
                file_dict[file_path] = file_dict[file_path] + "\n\n" + context.get_text()
        context = "\n\n".join(file_dict.values())

        print("Talking to the LLM")

        # Define the prompt template
        prompt_template = """
        You are an expert in Red Hat's US Benefits. Your role is to provide accurate and helpful information to employees regarding Red Hat's benefits plans and policies in the United States.

        When answering questions, please:
        - Provide clear and concise responses based the context provided only.
        - If the question is not related to Red Hat's US Benefits, politely decline to answer and explain that your expertise is relates only to Red Hat US Benifits.
        - Do not make assumptions or speculate beyond the context provided.
        - If there is any Not Safe for Work Content in the question, decline it to answer politely and do not use any explicit content.
        - Maintain a professional and helpful tone.

        Here is the context to consider when answering the question:
        {context}

        ### QUESTION: 
        {question}
        ### ANSWER:
        """

        # Create the prompt template instance
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        response = llm_chain.invoke({"context": context, "question": query})
        # print("==================================================================")
        # print("RAW RESPONSE:")
        # print(response)
        # print("==================================================================")
        # print("GENERATED HYPOTHETICAL DOCUMENT:")
        # print(generated_document)
        # print("==================================================================")
        # print("QUESTION:")
        # print(response["question"])
        # print("==================================================================")
        # print("RESPONSE:")
        # print(response['text'])
        # print("==================================================================")
        return response