from src.chatbot import create_chatbot_graph, run_chatbot

def main():
    print("Starting LangGraph Chatbot...")
    graph = create_chatbot_graph()
    run_chatbot(graph)


if __name__ == "__main__":
    main()
