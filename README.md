## Estructuras de Datos 

## Contenido
- `Stack` (LIFO) sobre lista.
- `Queue` (FIFO) con `collections.deque`.
- `SinglyLinkedList` con `push_front/pop_front/find/iter`.
- `BST` (árbol binario de búsqueda **sin** balanceo) con `insert/search/inorder`.
- `MinHeap` (envoltura mínima sobre `heapq`).
- `Graph` (lista de adyacencia) con `bfs`, `dfs`, `shortest_path_unweighted`.

Archivo principal: **`structures.py`** con CLI (`--demo` y `--bench`).

## Requisitos
- Python 3.10+
- (Opcional) `matplotlib` para graficar el benchmark.

> Este módulo no necesita `numpy`/`pandas`. En **Regresiones** y **Estadística** sí se usan.

## Cómo usar

Desde `data_structures/`:

```bash
# Demo rápida: muestra operaciones básicas en consola
python structures.py --demo

# Benchmark: guarda CSV y PNG en ./figs
python structures.py --bench
```

Artefactos generados por `--bench`:
- `figs/bench_times.csv` — tiempos de inserción/búsqueda para list/set/BST.
- `figs/bench_bar.png` — gráfico de barras (si `matplotlib` está instalado).

## Complejidades típicas (resumen)
| Estructura | Inserción | Búsqueda | Eliminación |
|---|---|---|---|
| `list` (append) | Amort. O(1) | O(n) | O(1) al final / O(n) general |
| `set` | Prom. O(1) | Prom. O(1) | Prom. O(1) |
| `BST` sin balanceo | Prom. O(log n), peor O(n) | Prom. O(log n), peor O(n) | Prom. O(log n), peor O(n) |
| `heapq` (MinHeap) | O(log n) (`push`) | — | O(log n) (`pop`) |
| `Queue`/`Stack` | O(1) | — | O(1) |

## Tips de compatibilidad (VS Code / Windows)
- Asegúrate de usar **Python 3.10+** (Ctrl+Shift+P → *Python: Select Interpreter*).
- Guarda el archivo como **UTF-8** y con saltos de línea **LF** (barra inferior de VS Code).
- Evita comillas “curvas” (usa comillas simples `'` o dobles `"` ASCII).
- Si Pylance sigue marcando errores de sintaxis, vuelve a pegar desde este README o reabre el archivo.

## Ejemplo rápido (Graph)

```python
from structures import Graph
g = Graph()
for u, v in [(1,2),(1,3),(2,4),(3,5),(4,6),(5,6)]:
    g.add_edge(u, v)

print(g.bfs(1))                         # [1, 2, 3, 4, 5, 6]
print(g.dfs(1))                         # e.g., [1, 3, 5, 6, 4, 2]
print(g.shortest_path_unweighted(1, 6)) # [1, 2, 4, 6] o [1, 3, 5, 6]
```

---
Sugerencias y mejoras son bienvenidas. La idea es que sea didáctico y fácil de correr en cualquier entorno.
