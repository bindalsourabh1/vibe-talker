import os

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.schema import SystemMessage
# from lang_graph.

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
@tool
def run_command(cmd: str):
    """
    You are on the powershell enviornment so execute commands accordingly
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    result = os.system(command=cmd)
    print(result)
    return result


llm = init_chat_model(model_provider='google-genai', model='gemini-2.0-flash')
llm_with_tool = llm.bind_tools(tools=[run_command])
tool_node = ToolNode(tools=[run_command])  # Provide actual tools if needed


def chatbot(state: State):
    system_prompt = SystemMessage(content="""
    You are an AI Coding assistant who takes an input from user and based on available
    tools you choose the correct tool and execute the commands.
    You can even execute commands and help user with the output of the command.
    Always make sure to keep your generated codes and files in chat_gpt/ folder. you can create one if not already there
                                  """)
    messages = llm_with_tool.invoke([system_prompt] + state["messages"])
    # assert len(messages.tool_calls) <= 1
    return {"messages": [messages]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# graph = graph_builder.compile()

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

