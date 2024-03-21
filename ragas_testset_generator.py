import os
from llama_index.core import SimpleDirectoryReader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

loader = SimpleDirectoryReader("hallucinatters_rag",num_files_limit=40)
documents = loader.load_data()
print("Loaded documents from rag for evaluation")


# generator with openai models
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OPENAI_API_KEY'] = ""
generator = TestsetGenerator.with_openai(generator_llm="gpt-3.5-turbo")

# generate testset
print("Generating evaluation test set")
testset = generator.generate_with_llamaindex_docs(documents, test_size=20, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
df=testset.to_pandas()
df.to_csv("questions.csv")
print("Test set generation completed")