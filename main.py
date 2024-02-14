import chainlit as cl
import genai

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

llm: ChatVertexAI = None
llm = ChatVertexAI(
    model_name='gemini-pro-vision',
    max_output_tokens=2048,
    temperature=0.4,
    top_p=1,
    top_k=32,
)


@cl.on_chat_start
async def _start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

    msg = cl.Message(
        content="Welcome to the Gemini chat!",
    )
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    print("history:",cl.user_session.get("message_history"))

    contents=[
        {
            'type': 'text',
            'text': message.content,
        },
    ]
    import base64
    images = [file for file in message.elements if "image" in file.mime]
    if len(images) > 0:
        with open(images[0].path, "rb") as f:
            encoded_data = base64.b64encode(f.read())
        image_content = {
            'type': 'image_url',
            'image_url': {
                "url": f"data:image/jpeg;base64,{encoded_data.decode('utf-8')}",
            }
        }
        # r_message['content'].append(image_content)
        contents.append(image_content)

    r_message = HumanMessage(content=contents)

    genned_message= llm.invoke([r_message])
    await cl.Message(
        content=genned_message.content,
    ).send()
