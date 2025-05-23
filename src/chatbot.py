import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
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
    graph_builder.set_entry_point("chatbot")

    # Add checkpointer
    memory = MemorySaver()
    
    graph = graph_builder.compile(checkpointer=memory)

    # Save graph visualization using graphviz
    try:
        import graphviz
        # Get the graphviz object from langgraph
        g = graph.get_graph()
        # Convert to graphviz format and save
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        
        # Add nodes and edges from the graph
        for node in g.nodes:
            dot.node(str(node), str(node))
        for edge in g.edges:
            dot.edge(str(edge[0]), str(edge[1]))
            
        # Save the visualization
        dot.render('chatbot_graph', format='png', cleanup=True)
        print("\nGraph visualization saved as 'chatbot_graph.png'")
    except Exception as e:
        print(f"\nCould not generate graph visualization: {e}")
        print("To enable graph visualization, install graphviz:")
        print("  pip install graphviz")
        print("  sudo apt-get install graphviz")

    return graph

def run_chatbot(graph):
    def format_message(message):
        role = "User" if message.type == "human" else "Assistant"
        return f"{role}: {message.content}"

    def show_memory_state(current_state):
        if not current_state or not current_state.values:
            print("\n[Memory State] No messages in memory yet\n")
            return
        
        messages = current_state.values.get('messages', [])
        print("\n[Memory State]")
        print(f"Total messages: {len(messages)}")
        print("\nConversation history:")
        for msg in messages:
            print(format_message(msg))
        print()

    def stream_graph_updates(user_input: str):
        # Simplified config with just thread_id
        config = {"configurable": {"thread_id": "main_thread"}}
        
        # Process the input and get response
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values"
        ):
            # Print each node's output as it comes in
            for node_name, node_output in event.items():
                if node_name == "messages":
                    # Check if there are any messages
                    if isinstance(node_output, list) and len(node_output) > 0:
                        # Get the last message
                        last_message = node_output[-1]
                        # If it's an AI message, print it
                        if hasattr(last_message, 'type') and last_message.type == "ai":
                            print("\nAssistant:", last_message.content)
                elif node_name == "tools":
                    print("\n[Using Tool: Tavily Search]")
                    for tool_call in node_output.get("tool_calls", []):
                        print(f"Tool Input: {tool_call.get('args', {})}")
                        print(f"Tool Output: {tool_call.get('result', '')}\n")

    print("\n[Memory Status]")
    print("Using MemorySaver - state will be lost when program exits")
    print("Type 'memory' to see conversation history, 'clear' to clear memory, or 'exit' to quit\n")

    # Simplified config
    config = {"configurable": {"thread_id": "main_thread"}}

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break
            elif user_input.lower() == "memory":
                current_state = graph.get_state(config)
                show_memory_state(current_state)
                continue
            elif user_input.lower() == "clear":
                graph.clear_state(config)
                print("\n[Memory cleared]\n")
                continue
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
