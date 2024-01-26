import chainlit as cl
import genai


@cl.on_chat_start
async def _start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    await cl.Message(
        content="Welcome to the Gemini chat!",
    ).send()


@cl.on_message
async def main(message: str):
    print("history:",cl.user_session.get("message_history"))
    genned_message = genai.run(message.content)
    await cl.Message(
        content=genned_message,
    ).send()
