import gradio as gr
from HYDE import Hyde


class Args:
    def __init__(self,persist_dir,model=None,chunk=1024,overlap=10,index_name='hallucinatters_rag_index'):
        self.persist_dir=persist_dir
        self.model=model
        self.chunk=chunk
        self.overlap=overlap
        self.index_name=index_name

def response(message, history):
    response=hyde.queryHyde(query=message)
    return response['text']


args=Args(persist_dir="ragpersist")
hyde=Hyde(args=args)

demo=gr.ChatInterface(
    fn=response,
    submit_btn="Ask"
    )

demo.launch(show_api=False,share=True,server_port=7860)