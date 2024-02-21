import chainlit as cl
import genai

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationChain

memory = ConversationBufferMemory(
    human_prefix="User", ai_prefix="Bot",
    memory_key="memory", return_messages=True,
)

@cl.on_chat_start
async def _start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "なんでもきいてね"}],
    )

    msg = cl.Message(
        content="Welcome to the Gemini chat!",
    )
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})

    kind: str = "gemini-pro"

    contents=[
        {
            'type': 'text',
            'text': message.content,
        },
    ]

    import base64
    image_files = [file for file in message.elements if "image" in file.mime]
    if len(image_files) > 0:
        kind = "gemini-pro-vision"
        with open(image_files[0].path, "rb") as f:
            encoded_data = base64.b64encode(f.read())
        image_content = {
            'type': 'image_url',
            'image_url': {
                "url": f"data:image/jpeg;base64,{encoded_data.decode('utf-8')}",
            }
        }
        contents.append(image_content)

    r_message = HumanMessage(content=contents)

    llm = genai.get_llm(kind)

    genned_message = llm.invoke([r_message])


    cl.user_session.set(
        "message_history",
        memory.load_memory_variables({})['memory'],
    )

    await cl.Message(
        content=genned_message.content,
    ).send()
