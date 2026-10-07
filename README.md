# Pre-entrega 5 — Agente de razonamiento cíclico con memoria persistente

Agente conversacional construido con **LangGraph**, que sigue el patrón **ReAct**
(Reason + Act): en vez de que el código decida cuándo usar una herramienta, es el
propio modelo de lenguaje quien lo decide, en un ciclo que se repite tantas veces
como haga falta hasta llegar a una respuesta final. El agente tiene memoria
persistente en disco (SQLite), así que recuerda conversaciones anteriores aunque
se reinicie el programa.

## Qué resuelve este agente

Responde preguntas sobre pedidos de clientes, usando dos herramientas simuladas
(`tools.py`) en vez de una base de datos real: buscar el ID de un cliente por
nombre, y buscar la cantidad de pedidos/monto total gastado por ID. El agente
puede necesitar **más de un paso** para responder (por ejemplo: primero buscar el
ID por nombre, después buscar los pedidos con ese ID) — ese encadenamiento de
pasos es el "razonamiento cíclico" que da nombre a la consigna.

## Arquitectura

```
tools.py    -> Fase 1: el "contrato" de herramientas (@tool). El LLM elige
               cuál usar basándose únicamente en el docstring de cada una.

graph.py    -> Fases 2 y 3: el StateGraph del agente.
               - Nodo "agente": le pasa la conversación al LLM (con las
                 herramientas ya vinculadas vía bind_tools) y deja que decida.
               - Nodo "tools" (ToolNode): ejecuta las herramientas que el LLM
                 pidió usar.
               - Arista condicional (tools_condition): si el LLM pidió una
                 herramienta, va a "tools"; si no, el grafo termina.
               - Arista "tools" -> "agente": cierra el ciclo, para que el LLM
                 razone de nuevo con el resultado de la herramienta.
               - Un recursion_limit evita que el ciclo se vuelva infinito.
               - El checkpointer (inyectado como parámetro) es lo que le da
                 memoria persistente al agente, indexada por thread_id.

main.py     -> Fase 4: script de ejecución. Corre 3 escenarios de prueba sobre
               el grafo real (con AsyncSqliteSaver, memoria en disco) y guarda
               la traza completa de cada uno como JSON en logs/.

tests/      -> Suite de tests automatizados. Ninguno necesita una API key real:
               usan un modelo "falso" (guionado) y un checkpointer en memoria
               para probar toda la lógica del agente de forma determinística.
```

### El ciclo ReAct, visualmente

```
   entrada del usuario
          |
          v
     [ agente ]  <----------------+
          |                       |
   ¿pidió una herramienta?        |
      /          \                |
    sí            no              |
     |             |              |
     v             v              |
  [ tools ]       FIN             |
     |                            |
     +----------------------------+
   (vuelve a "agente" con el resultado)
```

## Instalación

1. Crear y activar un entorno virtual con **Python 3.12 o 3.13**:

   ```bash
   python -m venv venv
   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   # Linux / Mac
   source venv/bin/activate
   ```

2. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Copiar `.env.example` a `.env` y completar la API key del proveedor que
   vayas a usar:

   ```bash
   cp .env.example .env    # en Windows: copy .env.example .env
   ```

### Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `LLM_PROVIDER` | `openai` o `anthropic` — qué proveedor de chat usar | `openai` |
| `OPENAI_API_KEY` | API key de OpenAI (si `LLM_PROVIDER=openai`) | — |
| `OPENAI_MODEL` | Modelo de OpenAI a usar | `gpt-4o-mini` |
| `ANTHROPIC_API_KEY` | API key de Anthropic (si `LLM_PROVIDER=anthropic`) | — |
| `ANTHROPIC_MODEL` | Modelo de Anthropic a usar | `claude-3-5-sonnet-20241022` |
| `SQLITE_DB_PATH` | Archivo donde se guarda la memoria persistente del agente | `checkpoints.sqlite` |

No hace falta ninguna base de datos externa ni índice vectorial en esta
pre-entrega: tanto los "clientes" como los "pedidos" son datos simulados en
memoria dentro de `tools.py`, y la persistencia del agente es un archivo SQLite
local que se crea solo.

## Cómo correr el agente

```bash
python main.py
```

Esto va a:

1. Crear (si no existe) el archivo `checkpoints.sqlite` con la memoria del agente.
2. Correr 3 escenarios de prueba (ver más abajo), imprimiendo en consola cada
   pregunta y respuesta.
3. Guardar la traza completa de los 3 escenarios (incluyendo qué herramientas
   se llamaron, con qué argumentos, y qué devolvieron) en
   `logs/traza_<fecha>_<hora>.json`.

### Los 3 escenarios que corre `main.py`

1. **Razonamiento multi-paso + memoria conversacional**: pregunta por los
   pedidos de "Carla" (el agente necesita encadenar `buscar_cliente_id` →
   `buscar_pedidos`), y después una pregunta de seguimiento en el mismo
   `thread_id` ("¿qué ID usaste?"), para demostrar que el agente recuerda la
   conversación anterior.
2. **"Ciclo de Retorno" — nombre ambiguo**: pide el cliente cuyo nombre
   contiene la letra "a", que matchea a los 3 clientes simulados a la vez. El
   agente debe pedir una aclaración en vez de inventar cuál es.
3. **"Ciclo de Retorno" — cliente inexistente**: pregunta por el cliente con
   ID 999, que no existe. Mismo principio: ante el error de la herramienta, el
   agente debe reconocerlo y pedir aclaración, no inventar una respuesta.

## Decisiones de diseño

- **Inyección de dependencias en todos lados** (`construir_grafo(checkpointer,
  modelo=None)`, `invocar_agente(grafo, ...)`): tanto el checkpointer como el
  modelo de lenguaje se reciben por parámetro en vez de crearse siempre adentro
  de la función. Esto es lo que permite testear toda la lógica del grafo (el
  ciclo, el ruteo condicional, la memoria) con un modelo falso y un
  checkpointer en memoria, sin gastar ni un token ni necesitar ninguna API key.
- **El nodo de herramientas se llama literalmente `"tools"`**: es el nombre que
  `tools_condition` (de `langgraph.prebuilt`) busca por default para rutear ahí
  cuando el último mensaje tiene `tool_calls`. Mantener el nombre convencional
  evita tener que pasar un `path_map` manual.
- **`recursion_limit` explícito** (`RECURSION_LIMIT = 10` en `graph.py`): como
  quien decide cuántas vueltas da el ciclo es el LLM y no el código, este techo
  evita el error común de "Bucles Infinitos" que señala la consigna.
- **Docstrings extensos en cada `@tool`** (`tools.py`): el LLM elige qué
  herramienta usar basándose únicamente en su docstring, así que son
  deliberadamente detallados sobre cuándo usar cada una y qué significa cada
  campo de la respuesta — para evitar el error común de "Descripciones Vagas".
- **El `SYSTEM_PROMPT` instruye explícitamente no inventar datos** ante un
  error de herramienta (cliente ambiguo o inexistente), y en cambio pedir una
  aclaración — esto es lo que hace posible el "Ciclo de Retorno" que pide la
  consigna.
- **`AsyncSqliteSaver` en producción, `MemorySaver` en los tests**: mismo
  checkpointer *interface*, pero en producción persiste en disco (memoria real
  entre ejecuciones) y en los tests vive solo en RAM (rápido, sin tocar el
  filesystem).
- **`recortar_historial_para_el_modelo` evita el "Estado Sucio"**: el estado
  del grafo (lo que guarda el checkpointer) acumula todos los mensajes para
  siempre -- eso es necesario para la memoria persistente y no se toca. Pero
  mandarle al LLM una conversación que no para de crecer sale caro y degrada
  la calidad de las respuestas. Esta función acota a los últimos
  `MAX_MENSAJES_EN_CONTEXTO` (20) mensajes lo que efectivamente se le envía al
  modelo en cada turno, usando `trim_messages` de LangChain con
  `start_on="human"` para no cortar justo en medio de un llamado a herramienta
  y dejar un `ToolMessage` sin su `AIMessage` correspondiente (lo cual
  rompería la llamada a la API).
- **Tipado estático y `asyncio` en todo el código**: tanto las herramientas
  (`tools.py`) como las funciones del grafo (`graph.py`) y del script de
  ejecución (`main.py`) usan type hints (`Dict`, `list[BaseMessage]`,
  `Optional[Runnable]`, etc.) y `async def`/`await` de punta a punta -- las
  herramientas son async porque simulan una operación de I/O real (una
  consulta a una base de datos), y el grafo entero se invoca con `ainvoke`.

## Testing

Todos los tests son automatizados, no requieren ninguna API key real, y se
corren con:

```bash
pytest tests/ -v
```

| Archivo | Qué prueba |
|---|---|
| `tests/test_tools.py` | Las herramientas en sí (`buscar_cliente_id`, `buscar_pedidos`): casos de éxito, nombre parcial, no encontrado, ambiguo, y que los docstrings sean lo suficientemente descriptivos. |
| `tests/test_graph.py` | La estructura y el ciclo del grafo: razonamiento multi-paso (≥2 tool calls), corte inmediato sin herramientas, memoria dentro de un mismo `thread_id`, aislamiento entre `thread_id` distintos, y el recorte de historial que evita el "Estado Sucio" — todo con un modelo falso guionado y `MemorySaver`. |
| `tests/test_main.py` | La lógica propia de `main.py`: conversión de mensajes a JSON (`mensaje_a_dict`) y la orquestación de un escenario completo (`correr_escenario`), incluyendo el caso de nombre ambiguo. |

Resultado actual: **18/18 tests pasando.**

## Salida real de una ejecución

_Pendiente: se va a completar con la salida real de `python main.py` corrida
con una API key válida, más el contenido del JSON generado en `logs/`._

## Criterios de aceptación (consigna)

- [x] **Autonomía** — el agente decide por sí mismo cuándo llamar a una
      herramienta, sin rutas manuales `if/else`: la decisión es 100% del LLM
      vía `llm.bind_tools(TOOLS)` + `tools_condition` (`graph.py`).
- [x] **Ciclo de Retorno** — si una herramienta devuelve un error o
      información incompleta, el agente pide aclaraciones en vez de inventar
      (instruido en el `SYSTEM_PROMPT`, ejercitado en 2 de los 3 escenarios de
      `main.py` y cubierto por tests).
- [x] **Resiliencia de Estado** — con un `thread_id`, el agente recuerda
      interacciones previas de la misma sesión (checkpointer + `MessagesState`,
      probado en `tests/test_graph.py` y demostrado en el escenario 1 de
      `main.py`).
- [x] **Código limpio** — Python 3.12+, type hints en `tools.py`/`graph.py`/
      `main.py`, y `asyncio` de punta a punta (herramientas y grafo son
      `async`).

## Checklist de entrega (consigna)

- [x] Repo público (sin API keys; variables de entorno vía `.env` +
      `.env.example`, `.env` en `.gitignore`) con README de cómo levantar el
      entorno.
- [x] `StateGraph` que hereda de `MessagesState`, con nodo de modelo + nodo de
      herramientas y arista condicional (`tools_condition`).
- [x] Al menos 1 herramienta con `@tool` y docstring descriptivo (acá hay 2:
      `buscar_cliente_id`, `buscar_pedidos`).
- [x] Persistencia con `SqliteSaver` (`AsyncSqliteSaver`) + `thread_id`
      (recuerda la sesión).
- [x] Prueba con razonamiento multi-paso (la herramienta se invoca ≥2 veces) y
      `recursion_limit` definido (`RECURSION_LIMIT = 10`).
- [x] Ejemplo de traza de ejecución (`.json`) incluido en `logs/`.

## Rúbrica de evaluación (autoevaluación)

| Criterio | Peso | Evidencia |
|---|---|---|
| Arquitectura del Grafo y Gestión de Estado | 30% | `StateGraph` sobre `MessagesState` (`graph.py`), reducer `operator.add` para los mensajes, `recortar_historial_para_el_modelo` para que el estado no quede "sucio" de cara al LLM sin perder lo persistido. |
| Integración de Herramientas y Razonamiento Multi-paso | 35% | 2 herramientas `@tool` con docstrings descriptivos, ciclo `agente → tools → agente` probado con ≥2 invocaciones antes de responder, `recursion_limit` como techo de seguridad. |
| Persistencia con SqliteSaver | 20% | `AsyncSqliteSaver` indexado por `thread_id`; memoria probada tanto en tests (`MemorySaver`) como en el escenario 1 de `main.py` (pregunta de seguimiento sobre el mismo hilo). |
| Calidad del Repositorio y Entorno Profesional | 15% | `requirements.txt` con versiones pinneadas, `.gitignore` cubre `.env`/`venv`/`checkpoints.sqlite`, type hints + `asyncio` en todo el código, 18 tests automatizados sin API keys reales. |

## Salida real de una ejecución

_Pendiente: se va a completar con la salida real de `python main.py` corrida
con una API key válida, más el contenido del JSON generado en `logs/`._
