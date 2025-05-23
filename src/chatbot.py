import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

class State(TypedDict):
    messages: Annotated[list, add_messages]

def create_chatbot_graph():
    # Load environment variables from .env file
    load_dotenv()

    # Initialize the chat model
    llm = init_chat_model(
        model="o4-mini",
        model_provider="azure-openai",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    # Create a simple tool(Tavily)
    tool = TavilySearch(max_results=2)
    tools = [tool]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        # Use llm_with_tools instead of llm to enable tool usage
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    

    # Create and configure the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()

    # Save graph visualization
    try:
        import graphviz
        dot = graphviz.Digraph(comment='Chatbot Graph')
        dot.attr(rankdir='LR')
        dot.node('START', 'START')
        dot.node('chatbot', 'Chatbot')
        dot.node('tools', 'Tools')
        dot.edge('START', 'chatbot')
        dot.edge('chatbot', 'tools')
        dot.edge('tools', 'chatbot')
        dot.render('chatbot_graph', format='png', cleanup=True)
        print("\nGraph visualization saved as 'chatbot_graph.png'")
    except Exception as e:
        print(f"\nCould not generate graph visualization: {e}")
        print("To enable graph visualization, install graphviz:")
        print("  pip install graphviz")
        print("  sudo apt-get install graphviz")

    return graph

def run_chatbot(graph):
    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for node, value in event.items():
                if node == "tools":
                    print("\n[Using Tool: Tavily Search]")
                    for tool_call in value.get("tool_calls", []):
                        print(f"Tool Input: {tool_call.get('args', {})}")
                        print(f"Tool Output: {tool_call.get('result', '')}\n")
                elif node == "chatbot":
                    print("Assistant: ", value["messages"][-1].content)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
