"""
tools.py
--------
Fase 1 de la consigna: El Contrato de Herramientas. Cada función
decorada con @tool es una herramienta que el agente puede elegir usar
-- el LLM decide CUÁL usar y con qué argumentos basándose ÚNICAMENTE en
el docstring, así que son deliberadamente descriptivos y explican
cuándo usar cada una.

La "base de datos" es un diccionario en memoria: la consigna pide
simular una operación de base de datos o búsqueda técnica, no conectar
una real.
"""

import asyncio
from typing import Dict, List

from langchain_core.tools import tool

# --- "Base de datos" simulada ---
_CLIENTES: Dict[int, str] = {
    102: "Carla Fernández",
    103: "Martín Gómez",
    104: "Lucía Paredes",
}

_PEDIDOS_POR_CLIENTE: Dict[int, Dict] = {
    102: {"pedidos": 3, "total": 14500},
    103: {"pedidos": 1, "total": 2200},
    104: {"pedidos": 5, "total": 31200},
}


@tool
async def buscar_cliente_id(nombre: str) -> Dict:
    """Busca el ID interno de un cliente a partir de su nombre o parte de él.

    Usá esta herramienta cuando el usuario mencione a un cliente POR
    NOMBRE (por ejemplo "Carla Fernández" o solo "Carla"), en vez de
    por su número de cliente. La búsqueda no distingue mayúsculas de
    minúsculas y encuentra coincidencias parciales.

    Args:
        nombre: El nombre (completo o parcial) del cliente a buscar.

    Returns:
        Un diccionario con "cliente_id" y "nombre" si se encontró
        exactamente un cliente que coincide. Si no se encontró ninguno,
        o si se encontraron varios (nombre ambiguo), devuelve en cambio
        la clave "error" con una descripción del problema -- en ese
        caso, no inventes un cliente_id: pedile una aclaración al
        usuario o intentá corregir la búsqueda.
    """
    await asyncio.sleep(0)  # simula una operación async de I/O real (ej. una consulta a una DB)

    busqueda = nombre.strip().lower()
    coincidencias = [
        (cid, nom) for cid, nom in _CLIENTES.items() if busqueda in nom.lower()
    ]

    if not coincidencias:
        return {"error": f"No se encontró ningún cliente cuyo nombre contenga '{nombre}'."}

    if len(coincidencias) > 1:
        nombres = ", ".join(nom for _, nom in coincidencias)
        return {
            "error": (
                f"'{nombre}' es ambiguo: coincide con varios clientes ({nombres}). "
                "Pedile al usuario que aclare a cuál se refiere."
            )
        }

    cliente_id, nombre_completo = coincidencias[0]
    return {"cliente_id": cliente_id, "nombre": nombre_completo}


@tool
async def buscar_pedidos(cliente_id: int) -> Dict:
    """Busca la cantidad de pedidos y el monto total gastado por un cliente, dado su ID numérico.

    Usá esta herramienta cuando ya tengas el ID NUMÉRICO del cliente
    (porque el usuario lo dio directamente, o porque lo obtuviste antes
    con buscar_cliente_id). No inventes un cliente_id que no tengas: si
    no lo tenés, buscalo primero con buscar_cliente_id, o pedíselo al
    usuario.

    Args:
        cliente_id: El identificador numérico del cliente.

    Returns:
        Un diccionario con "pedidos" (cantidad de pedidos) y "total"
        (monto total gastado en esos pedidos) si el cliente existe. Si
        el cliente_id no está registrado, devuelve en cambio la clave
        "error" -- en ese caso, no inventes un número: pedile una
        aclaración al usuario o revisá si te equivocaste de ID.
    """
    await asyncio.sleep(0)

    datos = _PEDIDOS_POR_CLIENTE.get(cliente_id)
    if datos is None:
        return {"error": f"No se encontró ningún cliente con ID {cliente_id}."}

    return {"pedidos": datos["pedidos"], "total": datos["total"]}


# Lista de herramientas que se le pasan al agente (bind_tools / ToolNode).
TOOLS: List = [buscar_cliente_id, buscar_pedidos]
