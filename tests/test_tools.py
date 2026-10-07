"""
tests/test_tools.py
---------------------
Prueba la LÓGICA de las herramientas directamente (sin pasar por
ningún LLM ni API real): cada @tool se invoca con `.ainvoke(...)`,
exactamente como lo haría el ToolNode del grafo, pero fuera del
agente -- así se valida el contrato de cada herramienta de forma
aislada y reproducible.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import buscar_cliente_id, buscar_pedidos


@pytest.mark.asyncio
async def test_buscar_cliente_id_encuentra_por_nombre_completo():
    resultado = await buscar_cliente_id.ainvoke({"nombre": "Carla Fernández"})
    assert resultado == {"cliente_id": 102, "nombre": "Carla Fernández"}


@pytest.mark.asyncio
async def test_buscar_cliente_id_encuentra_por_nombre_parcial_sin_distinguir_mayusculas():
    resultado = await buscar_cliente_id.ainvoke({"nombre": "carla"})
    assert resultado == {"cliente_id": 102, "nombre": "Carla Fernández"}


@pytest.mark.asyncio
async def test_buscar_cliente_id_no_encontrado_devuelve_error():
    resultado = await buscar_cliente_id.ainvoke({"nombre": "Nombre Que No Existe"})
    assert "error" in resultado


@pytest.mark.asyncio
async def test_buscar_cliente_id_ambiguo_devuelve_error():
    # "a" matchea a los 3 clientes de la base simulada -> ambiguo a propósito
    resultado = await buscar_cliente_id.ainvoke({"nombre": "a"})
    assert "error" in resultado
    assert "ambiguo" in resultado["error"]


@pytest.mark.asyncio
async def test_buscar_pedidos_devuelve_los_datos_esperados_del_ejemplo_de_la_consigna():
    # Este es literalmente el ejemplo de traza que da la consigna:
    # buscar_pedidos(cliente_id=102) -> {"pedidos": 3, "total": 14500}
    resultado = await buscar_pedidos.ainvoke({"cliente_id": 102})
    assert resultado == {"pedidos": 3, "total": 14500}


@pytest.mark.asyncio
async def test_buscar_pedidos_cliente_inexistente_devuelve_error():
    resultado = await buscar_pedidos.ainvoke({"cliente_id": 999})
    assert "error" in resultado
    assert "999" in resultado["error"]


@pytest.mark.asyncio
async def test_herramientas_tienen_docstrings_no_vacios():
    """Guarda contra el error más común de la consigna ('Descripciones
    Vagas'): las herramientas tienen que tener docstrings sustanciales,
    porque son lo único que el LLM usa para decidir cuál usar."""
    for herramienta in (buscar_cliente_id, buscar_pedidos):
        assert herramienta.description and len(herramienta.description) > 40
