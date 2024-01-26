
import os

from langchain_google_vertexai import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm :dict[str, ChatVertexAI|None] = {
    "gemini-pro": None,
    "gemini-pro-vision": None,
}

memory = ConversationBufferMemory()
_debug: bool = 'DEBUG' in os.environ

def _init():
    default_params: dict = {
        "model_name": "",
        "temperature": 0.6,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40,
        }
    p = default_params
    return p
    
def get_llm(kind: str = 'gemini-pro') -> ConversationChain:
    global llm, memory
    if kind in llm:
        p = _init()
        p["model_name"] = kind
        llm[kind] = ChatVertexAI(**p)
        print(f"Generated LLM:{kind}")

    chat_model = ConversationChain(
        llm=llm[kind],
        verbose=True,
        memory=memory,
    )
    
    return chat_model

def run(text_string: str) -> str:
    response = get_llm().predict(input=text_string)
    return response
