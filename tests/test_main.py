"""
tests/test_main.py
---------------------
Prueba la lógica propia de main.py que NO depende de tener un LLM o una
API key real: la conversión de mensajes a dict serializable
(`mensaje_a_dict`) y la orquestación de un escenario completo
(`correr_escenario`), usando el mismo patrón de modelo falso y
checkpointer en memoria que tests/test_graph.py.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import construir_grafo
from main import correr_escenario, mensaje_a_dict
from tests.test_graph import ModeloFalso


def test_mensaje_a_dict_mensaje_humano_simple():
    resultado = mensaje_a_dict(HumanMessage(content="hola"))
    assert resultado == {"tipo": "HumanMessage", "contenido": "hola"}


def test_mensaje_a_dict_respuesta_con_tool_calls():
    mensaje = AIMessage(
        content="",
        tool_calls=[{"name": "buscar_cliente_id", "args": {"nombre": "Carla"}, "id": "call_1"}],
    )
    resultado = mensaje_a_dict(mensaje)

    assert resultado["tipo"] == "AIMessage"
    assert resultado["tool_calls"] == [
        {"nombre": "buscar_cliente_id", "argumentos": {"nombre": "Carla"}, "id": "call_1"}
    ]


def test_mensaje_a_dict_resultado_de_herramienta():
    mensaje = ToolMessage(
        content='{"cliente_id": 102, "nombre": "Carla Fernández"}',
        name="buscar_cliente_id",
        tool_call_id="call_1",
    )
    resultado = mensaje_a_dict(mensaje)

    assert resultado["tipo"] == "ToolMessage"
    assert resultado["herramienta"] == "buscar_cliente_id"
    assert resultado["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_correr_escenario_arma_la_traza_completa_de_dos_preguntas():
    """Reproduce, con un modelo falso, el escenario 1 de main.py: una
    pregunta que dispara 2 tool_calls y una de seguimiento que usa la
    memoria del mismo thread_id."""
    modelo_falso = ModeloFalso(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "buscar_cliente_id", "args": {"nombre": "Carla"}, "id": "c1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "c2"}],
            ),
            AIMessage(content="Carla tuvo 3 pedidos por un total de $14.500."),
            AIMessage(content="Usé el ID de cliente 102."),
        ]
    )
    grafo = construir_grafo(checkpointer=MemorySaver(), modelo=modelo_falso)

    traza = await correr_escenario(
        grafo,
        nombre="escenario de prueba",
        preguntas=[
            "¿Cuántos pedidos hizo Carla y cuál fue el total?",
            "¿Qué ID de cliente usaste?",
        ],
        thread_id="test-escenario",
    )

    assert traza["escenario"] == "escenario de prueba"
    assert traza["thread_id"] == "test-escenario"
    assert len(traza["preguntas"]) == 2

    # La traza completa tiene que ser serializable a JSON (lista de dicts).
    tipos = [m["tipo"] for m in traza["traza_completa"]]
    assert tipos.count("ToolMessage") == 2
    assert tipos.count("HumanMessage") == 2
    assert "Usé el ID de cliente 102." in traza["traza_completa"][-1]["contenido"]


@pytest.mark.asyncio
async def test_correr_escenario_cliente_ambiguo_no_inventa_datos():
    """Reproduce el escenario 2 ('Ciclo de Retorno'): la herramienta
    devuelve un error de ambigüedad y el modelo (acá, guionado para
    simular una respuesta correcta) debe pedir una aclaración en vez de
    devolver un número inventado."""
    modelo_falso = ModeloFalso(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "buscar_cliente_id", "args": {"nombre": "a"}, "id": "c1"}],
            ),
            AIMessage(
                content="Ese nombre es ambiguo, coincide con varios clientes. ¿Podés aclarar a cuál te referís?"
            ),
        ]
    )
    grafo = construir_grafo(checkpointer=MemorySaver(), modelo=modelo_falso)

    traza = await correr_escenario(
        grafo,
        nombre="cliente ambiguo",
        preguntas=["Buscá al cliente cuyo nombre contiene la letra 'a'."],
        thread_id="test-ambiguo",
    )

    mensajes_herramienta = [m for m in traza["traza_completa"] if m["tipo"] == "ToolMessage"]
    assert len(mensajes_herramienta) == 1
    assert "ambigu" in traza["traza_completa"][-1]["contenido"].lower()
