import chainlit as cl
import genai

from langchain.globals import set_debug, set_verbose

set_verbose(True)
set_debug(True)

from langchain.schema import messages_from_dict, messages_to_dict

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationChain

memory = ConversationBufferMemory(
    human_prefix="User", ai_prefix="Bot",
    memory_key="history",
    verbose=True,
    return_messages=True,
)

@cl.on_chat_start
async def _start():
    message: str = "なんでもきいてね！でも記憶力はないですよ"
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": message}],
    )

    msg = cl.Message(
        content=message,
    )
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    # history.append({"role": "user", "content": message.content})

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

    buffer = memory.load_memory_variables({})
    pre = messages_to_dict(buffer['history'])
    memory_data = memory.load_memory_variables({})['history']
    print(memory_data)
    genned_message = llm.invoke([
        r_message,
    ])

    memory.chat_memory.add_user_message(r_message)
    memory.chat_memory.add_ai_message(genned_message.content)

    print(memory_data)

    cl.user_session.set(
        "message_history",
        memory.load_memory_variables({})['history'],
    )

    await cl.Message(
        content=genned_message.content,
    ).send()
