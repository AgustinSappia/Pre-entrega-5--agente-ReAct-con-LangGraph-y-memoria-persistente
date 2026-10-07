"""
tests/test_graph.py
----------------------
Prueba la ESTRUCTURA y el CICLO del grafo (nodos, aristas condicionales,
y persistencia con checkpointer) sin ningún LLM real: se inyecta un
modelo falso con respuestas "guionadas" (scripted), que en la primera
llamada pide usar una herramienta, en la segunda pide usar otra, y
recién en la tercera responde -- exactamente el escenario de
"razonamiento multi-paso" que pide la consigna (la herramienta se
invoca al menos dos veces). El checkpointer es `MemorySaver` (en
memoria, sin tocar disco), para probar la lógica de persistencia sin
pagar el costo de I/O de un archivo SQLite real en cada test.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import construir_grafo, invocar_agente, recortar_historial_para_el_modelo


class ModeloFalso:
    """Sustituto de un ChatModel real: en vez de razonar de verdad,
    devuelve una respuesta de una lista pre-armada, una por cada vez
    que se lo invoca. Así se puede "guionar" un ciclo ReAct completo
    de forma determinística para el test."""

    def __init__(self, respuestas):
        self._respuestas = list(respuestas)
        self._llamadas = 0

    def bind_tools(self, tools):
        # El grafo real llama a .bind_tools(TOOLS) antes de usar el
        # modelo; acá no hace falta hacer nada con `tools`, el modelo
        # falso ya tiene sus respuestas decididas de antemano.
        return self

    async def ainvoke(self, mensajes):
        indice = min(self._llamadas, len(self._respuestas) - 1)
        respuesta = self._respuestas[indice]
        self._llamadas += 1
        return respuesta


@pytest.mark.asyncio
async def test_el_grafo_ejecuta_dos_llamadas_a_herramientas_antes_de_responder():
    """Escenario central de la consigna: 'razonamiento multi-paso',
    la herramienta se invoca >= 2 veces antes de la respuesta final."""
    respuestas_guionadas = [
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_cliente_id", "args": {"nombre": "Carla"}, "id": "call_1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "call_2"}],
        ),
        AIMessage(content="El cliente 102 tuvo 3 pedidos por un total de $14.500."),
    ]
    modelo_falso = ModeloFalso(respuestas_guionadas)
    grafo = construir_grafo(checkpointer=MemorySaver(), modelo=modelo_falso)

    resultado = await invocar_agente(
        grafo,
        "¿Cuántos pedidos tuvo Carla y cuál fue el total?",
        thread_id="test-multi-paso",
    )

    mensajes = resultado["messages"]

    # Se consumieron las 3 respuestas guionadas -> el modelo fue
    # invocado 3 veces (2 decisiones de usar herramienta + 1 respuesta final).
    assert modelo_falso._llamadas == 3

    mensajes_de_herramienta = [m for m in mensajes if isinstance(m, ToolMessage)]
    assert len(mensajes_de_herramienta) == 2

    ultimo_mensaje = mensajes[-1]
    assert "3 pedidos" in ultimo_mensaje.content
    assert "14.500" in ultimo_mensaje.content


@pytest.mark.asyncio
async def test_el_grafo_termina_sin_herramientas_si_el_modelo_no_las_pide():
    modelo_falso = ModeloFalso([AIMessage(content="Hola, ¿en qué te puedo ayudar?")])
    grafo = construir_grafo(checkpointer=MemorySaver(), modelo=modelo_falso)

    resultado = await invocar_agente(grafo, "Hola", thread_id="test-sin-herramientas")

    assert modelo_falso._llamadas == 1
    assert resultado["messages"][-1].content == "Hola, ¿en qué te puedo ayudar?"


@pytest.mark.asyncio
async def test_el_checkpointer_recuerda_el_historial_con_el_mismo_thread_id():
    modelo_falso = ModeloFalso(
        [
            AIMessage(content="El cliente 102 tuvo 3 pedidos."),
            AIMessage(content="Su último pedido fue el de mayor monto del total."),
        ]
    )
    checkpointer = MemorySaver()
    grafo = construir_grafo(checkpointer=checkpointer, modelo=modelo_falso)

    primer_resultado = await invocar_agente(
        grafo, "¿Cuántos pedidos tuvo el cliente 102?", thread_id="misma-sesion"
    )
    assert len(primer_resultado["messages"]) == 2  # 1 humano + 1 IA

    segundo_resultado = await invocar_agente(
        grafo, "¿Y el último?", thread_id="misma-sesion"
    )
    # El historial se ACUMULA: ahora tiene los 2 mensajes anteriores
    # más los 2 nuevos (pregunta + respuesta) = 4.
    assert len(segundo_resultado["messages"]) == 4
    assert "¿Cuántos pedidos tuvo el cliente 102?" in segundo_resultado["messages"][0].content


@pytest.mark.asyncio
async def test_el_checkpointer_no_mezcla_threads_distintos():
    modelo_falso = ModeloFalso([AIMessage(content="respuesta para cualquier pregunta")])
    checkpointer = MemorySaver()
    grafo = construir_grafo(checkpointer=checkpointer, modelo=modelo_falso)

    await invocar_agente(grafo, "pregunta en el hilo A", thread_id="hilo-a")
    resultado_hilo_b = await invocar_agente(grafo, "pregunta en el hilo B", thread_id="hilo-b")

    # El hilo B no debería tener memoria de lo que pasó en el hilo A.
    assert len(resultado_hilo_b["messages"]) == 2


def test_recortar_historial_no_toca_nada_si_ya_entra_en_el_limite():
    mensajes = [HumanMessage(content="hola"), AIMessage(content="hola, ¿en qué ayudo?")]

    resultado = recortar_historial_para_el_modelo(mensajes, max_mensajes=10)

    assert resultado == mensajes


def test_recortar_historial_se_queda_con_los_ultimos_mensajes_si_se_pasa_del_limite():
    """Prueba directa del 'Errores Comunes a Evitar -> Estado Sucio':
    con una conversación larga (30 mensajes), el historial que se le
    manda al modelo queda acotado -- sin que esto afecte lo que el
    checkpointer guarda (eso se prueba en los tests de arriba, donde el
    estado completo sí se acumula sin límite)."""
    mensajes = []
    for i in range(15):
        mensajes.append(HumanMessage(content=f"pregunta {i}"))
        mensajes.append(AIMessage(content=f"respuesta {i}"))

    resultado = recortar_historial_para_el_modelo(mensajes, max_mensajes=10)

    assert len(resultado) == 10
    # Tiene que empezar en un mensaje humano, no puede cortar justo en
    # medio de un intercambio y dejar una respuesta sin su pregunta.
    assert isinstance(resultado[0], HumanMessage)
    # Y tiene que ser la PARTE MÁS RECIENTE de la conversación.
    assert resultado[-1].content == "respuesta 14"
