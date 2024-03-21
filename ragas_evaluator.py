from HYDE import Hyde
from RAG import Rag
import argparse
import pandas as pd
from datasets import Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OPENAI_API_KEY'] = ""
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas import evaluate
import re


def evaluate_hyde(args):
    hyde=Hyde(args=args)
    df=pd.read_csv("questions.csv")
    questions=list(df["question"])
    
    responses=[hyde.queryHyde(q) for q in questions]
    answers=[]
    contexts=[]

    for r in responses:
        answer_text = re.sub(r'\s+', ' ', r['text'].replace('\n', ' ')).strip()
        context_text = re.sub(r'\s+', ' ', r['context'].replace('\n', ' ')).strip()
        answers.append(f'"{answer_text}"')  # Replace newline characters with a space
        contexts.append(f'"{context_text}"')
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    
    dataset_dict["ground_truth"] = list(df["ground_truth"])
    df_dataset = pd.DataFrame(dataset_dict)
    df_dataset.fillna("", inplace=True)
    float_columns = df_dataset.select_dtypes(include=["float"]).columns
    df_dataset[float_columns] = df_dataset[float_columns].astype(str)
    dataset_dict = df_dataset.to_dict(orient="list")
    ds = Dataset.from_dict(dataset_dict)
    ds.to_csv("hyde_reponses.csv")
    ds=Dataset.from_csv("hyde_reponses.csv")

    metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    ]

    result_hyde = evaluate(
    ds,
    metrics=metrics,
    )

    return_value = result_hyde
    result_hyde=result_hyde.to_pandas()
    result_hyde.to_csv("hyde_result.csv")
    return return_value


def evaluate_rag(args):
    rag=Rag(args=args)
    df=pd.read_csv("questions.csv")
    questions=list(df["question"])
    
    responses=[rag.queryRag(q) for q in questions]
    answers=[]
    contexts=[]

    for r in responses:
        answer_text = re.sub(r'\s+', ' ', r['text'].replace('\n', ' ')).strip()
        context_text = re.sub(r'\s+', ' ', r['context'].replace('\n', ' ')).strip()
        answers.append(f'"{answer_text}"')  # Replace newline characters with a space
        contexts.append(f'"{context_text}"')
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    
    dataset_dict["ground_truth"] = list(df["ground_truth"])
    df_dataset = pd.DataFrame(dataset_dict)
    df_dataset.fillna("", inplace=True)
    float_columns = df_dataset.select_dtypes(include=["float"]).columns
    df_dataset[float_columns] = df_dataset[float_columns].astype(str)
    dataset_dict = df_dataset.to_dict(orient="list")
    ds = Dataset.from_dict(dataset_dict)
    ds.to_csv("rag_reponses.csv")
    ds=Dataset.from_csv("rag_reponses.csv")

    metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    ]

    result_rag = evaluate(
    ds,
    metrics=metrics,
    )

    return_value = result_rag
    result_rag=result_rag.to_pandas()
    result_rag.to_csv("rag_result.csv")
    return return_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="embedding cli for task execution"
    )
    parser.add_argument("-p", "--persist-dir", help="Vector DB persist dir")
    parser.add_argument("-c", "--chunk",   default="1024", help="Chunk size for embedding")
    parser.add_argument("-l", "--overlap",   default="10", help="Chunk overlap for embedding")
    parser.add_argument("-i", "--index-name", default="hallucinatters_rag_index", help="Vector DB persist dir")
    parser.add_argument("-m", "--model", help="LLM model used for embeddings [local, llama2, or any other supported by llama_index]")

    # execute
    args = parser.parse_args()
    # rag_results = evaluate_rag(args)
    hyde_results = evaluate_hyde(args)
    # print("RAG results:")
    # print(rag_results)
    print("HYDE results:")
    print(hyde_results)
    