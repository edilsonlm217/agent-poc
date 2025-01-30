from langgraph.graph import StateGraph, START, END, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Definir estado
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # Usando add como reducer

# Configurar modelo Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7
)

# Nó de geração de resposta
def generate_response(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# Construir grafo
workflow = StateGraph(AgentState)

# Adicionar nós
workflow.add_node("generate", generate_response)

# Definir arestas
workflow.add_edge(START, "generate")
workflow.add_edge("generate", END)

# Adiciona persistência básica
memory = MemorySaver()

# Compilar grafo
graph = workflow.compile(checkpointer=memory)