"""
graph.py
--------
Fase 2 y 3 de la consigna: el StateGraph del agente (nodo del modelo +
nodo de herramientas conectados con una arista condicional, cerrando
un ciclo) y la configuración del checkpointer de persistencia.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

load_dotenv(override=True)

# Errores comunes a evitar - "Bucles Infinitos": como quien decide el
# flujo es el LLM (no nosotros), el grafo podría en teoría seguir
# llamando herramientas indefinidamente. Este techo lo evita.
RECURSION_LIMIT = 10

# Errores comunes a evitar - "Estado Sucio": el estado del grafo (lo que
# guarda el checkpointer) ACUMULA todos los mensajes para siempre -- eso
# es lo que le da memoria persistente al agente, y no lo tocamos. Pero
# mandarle al LLM una conversación que no para de crecer sale caro y
# degrada la calidad de las respuestas. Este techo acota cuántos
# mensajes RECIENTES se le mandan al modelo en cada turno (el historial
# completo sigue ahí, guardado, por si se necesita después).
MAX_MENSAJES_EN_CONTEXTO = 20

SYSTEM_PROMPT = (
    "Sos un asistente que responde preguntas sobre pedidos de clientes "
    "usando las herramientas disponibles. Nunca inventes datos de "
    "pedidos o clientes: solo usá lo que te devuelvan las herramientas. "
    "Si una herramienta devuelve un error (por ejemplo, 'cliente no "
    "encontrado' o un nombre ambiguo), no lo ignores ni sigas como si "
    "nada: intentá corregir el llamado si te parece razonable (por "
    "ejemplo, revisando el nombre que usaste), o si no podés "
    "resolverlo vos solo, pedile a la persona que aclare a quién se "
    "refiere, en vez de inventar una respuesta."
)


def _construir_modelo():
    """Mismo patrón factory de las pre-entregas anteriores: elegir el
    proveedor de chat según una variable de entorno."""
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        modelo = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=modelo, temperature=0)

    elif provider == "anthropic":
        modelo = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(model=modelo, temperature=0)

    raise ValueError(f"Proveedor no soportado: {provider}")


def recortar_historial_para_el_modelo(
    mensajes: list[BaseMessage], max_mensajes: int = MAX_MENSAJES_EN_CONTEXTO
) -> list[BaseMessage]:
    """Devuelve solo los últimos `max_mensajes` mensajes de `mensajes`.

    Importante: esto NO borra nada del estado persistido por el
    checkpointer -- el historial completo sigue acumulándose ahí
    siempre (eso es lo que pide 'Resiliencia de Estado'). Esta función
    se usa únicamente para armar la lista que se le manda al LLM en
    cada turno, así la conversación no crece sin límite de cara a la
    API (evita el error común de 'Estado Sucio').

    `start_on="human"` asegura que el recorte empiece en un mensaje del
    usuario, para no dejar un ToolMessage "huérfano" (sin el AIMessage
    con el tool_call que lo originó) al principio de la lista, lo cual
    rompería la llamada a la API.
    """
    if len(mensajes) <= max_mensajes:
        return mensajes

    return trim_messages(
        mensajes,
        max_tokens=max_mensajes,
        token_counter=lambda msgs: len(msgs) if isinstance(msgs, list) else 1,
        strategy="last",
        start_on="human",
        include_system=False,
    )


def construir_grafo(checkpointer, modelo: Optional[Runnable] = None) -> CompiledStateGraph:
    """Arma y compila el StateGraph del agente.

    Tanto `checkpointer` como `modelo` se reciben por parámetro, en vez
    de construirse siempre adentro de esta función -- el mismo patrón
    de inyección de dependencias de las pre-entregas anteriores. Acá es
    lo que permite testear la ESTRUCTURA y el CICLO del grafo (nodos,
    aristas, memoria) con un modelo falso y un checkpointer en memoria,
    sin ninguna API key real.
    """
    llm = modelo or _construir_modelo()
    llm_con_herramientas = llm.bind_tools(TOOLS)

    async def nodo_agente(state: MessagesState) -> dict:
        """Fase 2: nodo del modelo. Le antepone el system prompt a la
        conversación acumulada (sin persistirlo como parte del
        historial guardado, para no duplicarlo en cada recarga del
        checkpointer) y le pide al modelo, YA CON LAS HERRAMIENTAS
        VINCULADAS, que decida solo si responde directamente o si
        necesita llamar a alguna."""
        historial_acotado = recortar_historial_para_el_modelo(state["messages"])
        respuesta = await llm_con_herramientas.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *historial_acotado]
        )
        # Solo devolvemos el mensaje NUEVO: el reducer de MessagesState
        # (operator.add) es el que lo concatena al historial existente.
        return {"messages": [respuesta]}

    grafo = StateGraph(MessagesState)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("tools", ToolNode(TOOLS))

    grafo.set_entry_point("agente")
    # tools_condition mira el último mensaje: si tiene tool_calls, va
    # al nodo "tools"; si no, el grafo termina (END).
    grafo.add_conditional_edges("agente", tools_condition)
    # Esta arista es la que CIERRA EL CICLO: después de ejecutar una
    # herramienta, se vuelve siempre al modelo, para que razone de
    # nuevo con el resultado -- y decida si ya puede responder o si
    # necesita llamar a otra herramienta más.
    grafo.add_edge("tools", "agente")

    return grafo.compile(checkpointer=checkpointer)


async def invocar_agente(
    grafo: CompiledStateGraph,
    mensaje_usuario: str,
    thread_id: str,
    recursion_limit: int = RECURSION_LIMIT,
) -> dict:
    """Invoca el grafo con un mensaje nuevo del usuario, dentro de la
    conversación identificada por `thread_id` (Fase 3: persistencia).
    Siempre fija un recursion_limit explícito."""
    entrada = {"messages": [HumanMessage(content=mensaje_usuario)]}
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    return await grafo.ainvoke(entrada, config=config)
