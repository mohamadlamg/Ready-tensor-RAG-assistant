from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

load_dotenv()
os.getenv('TAVILY_API_KEY')

api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0.8,
               api_key=api_key)


class State(TypedDict):
    messages : Annotated[list,add_messages]

def get_tools():
    return[
        TavilySearch(max_results=3,search_depth='advanced')
    ]

def llm_node(State):
    tools = get_tools()
    # Pour permettre au llm  d'avoir accès aux outils
    llm_with_tools = llm.bind_tools(tools) 
    response = llm_with_tools.invoke(State['messages'])
    return {"messages": [response]} 

def tools_node(State):
    tools = get_tools()
    tool_registry = {tool.name : tool for tool in tools}

    last_messages = State["messages"][-1]
    tool_messages = []

    for tool_call in last_messages.tool_calls:
        tool = tool_registry[tool_call['name']]
        result = tool.invoke(tool_call['args'])

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id= tool_call['id']
            )
        )
    
    return{
        'messages':tool_messages
    }

def should_continue(State):
    last_message = State['messages'][-1]

    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "tools"
    else :
        return "end"


def create_agent():
    graph = StateGraph(State)

    graph.add_node("llm",llm_node)
    graph.add_node("tools",tools_node)

    graph.set_entry_point("llm")


    graph.add_conditional_edges("llm",should_continue,{"tools":"tools","end":END})
    graph.add_edge("tools","llm")

    return graph.compile()

agent = create_agent()

initial_state = {
    "messages":[
        SystemMessage(content="You are a helpful assistant with access to web search. Use the search tool when you need current information.")
        ,HumanMessage(content="What is the latest news about AI development in 2025 ?")
    ]
}

result = agent.invoke(initial_state)
print(result["messages"][-1].content)