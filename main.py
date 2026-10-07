"""
main.py
-------
Fase 4 de la consigna: script de ejecución del agente. Corre tres
escenarios de prueba sobre el grafo REAL (modelo de verdad + checkpointer
SQLite persistente en disco) y guarda la traza completa de cada uno
(incluyendo los tool_calls que pidió el modelo y los resultados que
devolvieron las herramientas) como JSON en logs/, tal como pide la
consigna como evidencia de ejecución incluida en el repo.

Escenarios:
  1. Razonamiento multi-paso + memoria: el ejemplo literal de la
     consigna (Carla -> buscar_cliente_id -> buscar_pedidos -> una
     respuesta que usa ambos resultados), seguido de una pregunta de
     seguimiento en el MISMO thread_id para demostrar que el agente
     recuerda la conversación anterior.
  2. "Ciclo de Retorno" con nombre ambiguo: le pedimos el cliente cuyo
     nombre contiene la letra "a", que matchea a los 3 clientes
     simulados. Un agente mal diseñado inventaría una respuesta; este
     agente, por el SYSTEM_PROMPT, debe pedir una aclaración.
  3. "Ciclo de Retorno" con cliente inexistente: pedimos el cliente con
     ID 999 (no existe en la base simulada), mismo principio.

Para correrlo hace falta una API key real en .env (ver .env.example).
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from graph import construir_grafo, invocar_agente

load_dotenv(override=True)

LOGS_DIR = Path(__file__).parent / "logs"
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.sqlite")


def mensaje_a_dict(mensaje: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje de LangChain a un dict serializable a JSON.

    Se queda con lo esencial para poder leer la traza después sin tener
    que volver a correr el agente: el tipo de mensaje, su contenido, y
    -si corresponde- los tool_calls que pidió el modelo o el resultado
    que devolvió una herramienta.
    """
    base = {
        "tipo": mensaje.__class__.__name__,
        "contenido": mensaje.content,
    }

    tool_calls = getattr(mensaje, "tool_calls", None)
    if tool_calls:
        base["tool_calls"] = [
            {"nombre": tc["name"], "argumentos": tc["args"], "id": tc["id"]}
            for tc in tool_calls
        ]

    if isinstance(mensaje, ToolMessage):
        base["herramienta"] = mensaje.name
        base["tool_call_id"] = mensaje.tool_call_id

    return base


async def correr_escenario(
    grafo: CompiledStateGraph, nombre: str, preguntas: list[str], thread_id: str
) -> dict[str, Any]:
    """Corre una o más preguntas seguidas sobre el mismo thread_id (así
    se puede probar memoria conversacional dentro de un mismo
    escenario) e imprime en consola cada intercambio. Devuelve un dict
    con la traza completa del thread, lista para guardar como JSON.
    """
    print(f"\n{'=' * 70}\nESCENARIO: {nombre}\n{'=' * 70}")

    resultado = None
    for pregunta in preguntas:
        print(f"\n>>> Usuario: {pregunta}")
        resultado = await invocar_agente(grafo, pregunta, thread_id=thread_id)
        print(f"<<< Agente: {resultado['messages'][-1].content}")

    return {
        "escenario": nombre,
        "thread_id": thread_id,
        "preguntas": preguntas,
        "traza_completa": [mensaje_a_dict(m) for m in resultado["messages"]],
    }


async def main():
    LOGS_DIR.mkdir(exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as checkpointer:
        grafo = construir_grafo(checkpointer=checkpointer)

        trazas = []

        trazas.append(
            await correr_escenario(
                grafo,
                nombre="Razonamiento multi-paso + memoria conversacional",
                preguntas=[
                    "¿Cuántos pedidos hizo Carla y cuál fue el total gastado?",
                    "¿Cuál fue el ID de cliente que usaste recién para averiguar eso?",
                ],
                thread_id="demo-multi-paso",
            )
        )

        trazas.append(
            await correr_escenario(
                grafo,
                nombre="Ciclo de Retorno: nombre ambiguo",
                preguntas=[
                    "Buscá al cliente cuyo nombre contiene la letra 'a' y decime sus pedidos."
                ],
                thread_id="demo-ambiguo",
            )
        )

        trazas.append(
            await correr_escenario(
                grafo,
                nombre="Ciclo de Retorno: cliente inexistente",
                preguntas=["¿Cuántos pedidos tiene el cliente con ID 999?"],
                thread_id="demo-inexistente",
            )
        )

    marca_de_tiempo = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archivo_salida = LOGS_DIR / f"traza_{marca_de_tiempo}.json"
    archivo_salida.write_text(
        json.dumps(trazas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n\nTraza completa de los {len(trazas)} escenarios guardada en: {archivo_salida}")


if __name__ == "__main__":
    asyncio.run(main())
