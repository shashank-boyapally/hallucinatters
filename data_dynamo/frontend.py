import gradio as gr
from llama_index.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.query_engine.pandas import PandasInstructionParser
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
import os


os.environ["OPENAI_API_KEY"] = "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"

import pandas as pd



def get_output(question, file):
    df = pd.read_csv(file)
    output_image="./output_graph.png"
    if os.path.exists(output_image):
        os.remove(output_image)
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas, Seaborn, Numpy and matplotlib.pyplot.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. If the code has a graph, the graph should be readable and use a wide figsize then add plt.savefig() with filename output_graph.png\n"
        "5. PRINT ONLY THE EXPRESSION.\n"
        "6. Do not quote the expression.\n"
    )

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`.\n"
        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )
    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n\n"
        "Response: "
    )


    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5)
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
    llm = OpenAI(model="gpt-3.5-turbo")

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link(
                "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
            ),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output",
            ),
        ]
    )
    # add link from response synthesis prompt to llm2
    qp.add_link("response_synthesis_prompt", "llm2")
    response = qp.run(
        query_str=question,
    )
    if os.path.exists(output_image):
        return response.message.content, "output_graph.png"

    return response.message.content, None

demo = gr.Interface(
    fn=get_output,
    inputs=["text",gr.File()],
    outputs=["text",gr.Image(width=1200)],
)

demo.launch()
