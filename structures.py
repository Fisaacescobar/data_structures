#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Estructuras de datos clásicas + demo/benchmark.
# Versión "modernizada y segura": mantiene tipado claro sin depender de __future__
# ni construcciones que a veces rompen el pegado en VS Code/Pylance.

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import argparse
import heapq
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Deque

# ---------------------------------------------------------------------
# Stack (LIFO)
# ---------------------------------------------------------------------
class Stack:
    def __init__(self) -> None:
        self._data: List[Any] = []

    def push(self, x: Any) -> None:
        self._data.append(x)

    def pop(self) -> Any:
        if not self._data:
            raise IndexError('pop from empty Stack')
        return self._data.pop()

    def peek(self) -> Any:
        if not self._data:
            raise IndexError('peek from empty Stack')
        return self._data[-1]

    def __len__(self) -> int:
        return len(self._data)

    def empty(self) -> bool:
        return len(self._data) == 0


# ---------------------------------------------------------------------
# Queue (FIFO) con deque
# ---------------------------------------------------------------------
class Queue:
    def __init__(self) -> None:
        self._dq: Deque[Any] = deque()

    def enqueue(self, x: Any) -> None:
        self._dq.append(x)

    def dequeue(self) -> Any:
        if not self._dq:
            raise IndexError('dequeue from empty Queue')
        return self._dq.popleft()

    def __len__(self) -> int:
        return len(self._dq)

    def empty(self) -> bool:
        return len(self._dq) == 0


# ---------------------------------------------------------------------
# Singly Linked List
# ---------------------------------------------------------------------
@dataclass
class _Node:
    value: Any
    next: Optional['_Node'] = None  # forward-ref como string para compatibilidad


class SinglyLinkedList:
    def __init__(self) -> None:
        self.head: Optional[_Node] = None
        self._n: int = 0

    def push_front(self, x: Any) -> None:
        self.head = _Node(x, self.head)
        self._n += 1

    def pop_front(self) -> Any:
        if self.head is None:
            raise IndexError('pop from empty list')
        val = self.head.value
        self.head = self.head.next
        self._n -= 1
        return val

    def find(self, x: Any) -> bool:
        cur = self.head
        while cur is not None:
            if cur.value == x:
                return True
            cur = cur.next
        return False

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        cur = self.head
        while cur is not None:
            yield cur.value
            cur = cur.next


# ---------------------------------------------------------------------
# Binary Search Tree (sin balanceo)
# ---------------------------------------------------------------------
@dataclass
class _BSTNode:
    key: Any
    left: Optional['_BSTNode'] = None
    right: Optional['_BSTNode'] = None


class BST:
    def __init__(self) -> None:
        self.root: Optional[_BSTNode] = None
        self._n: int = 0

    def insert(self, key: Any) -> None:
        if self.root is None:
            self.root = _BSTNode(key)
            self._n += 1
            return
        cur = self.root
        while True:
            if key == cur.key:
                return  # sin duplicados
            elif key < cur.key:
                if cur.left is None:
                    cur.left = _BSTNode(key)
                    self._n += 1
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = _BSTNode(key)
                    self._n += 1
                    return
                cur = cur.right

    def search(self, key: Any) -> bool:
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return True
            if key < cur.key:
                cur = cur.left
            else:
                cur = cur.right
        return False

    def inorder(self) -> List[Any]:
        res: List[Any] = []

        def _dfs(n: Optional[_BSTNode]) -> None:
            if n is None:
                return
            _dfs(n.left)
            res.append(n.key)
            _dfs(n.right)

        _dfs(self.root)
        return res

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------
# MinHeap (wrapper sobre heapq)
# ---------------------------------------------------------------------
class MinHeap:
    def __init__(self, data: Optional[Iterable[Any]] = None) -> None:
        self._h: List[Any] = []
        if data is not None:
            self._h = list(data)
            heapq.heapify(self._h)

    def push(self, x: Any) -> None:
        heapq.heappush(self._h, x)

    def pop(self) -> Any:
        if not self._h:
            raise IndexError('pop from empty MinHeap')
        return heapq.heappop(self._h)

    def peek(self) -> Any:
        if not self._h:
            raise IndexError('peek from empty MinHeap')
        return self._h[0]

    def __len__(self) -> int:
        return len(self._h)


# ---------------------------------------------------------------------
# Grafo (lista de adyacencia) + BFS/DFS y camino mínimo sin pesos
# ---------------------------------------------------------------------
class Graph:
    def __init__(self) -> None:
        self._adj: Dict[Any, Set[Any]] = {}

    def add_edge(self, u: Any, v: Any, undirected: bool = True) -> None:
        if u not in self._adj:
            self._adj[u] = set()
        self._adj[u].add(v)
        if undirected:
            if v not in self._adj:
                self._adj[v] = set()
            self._adj[v].add(u)
        else:
            if v not in self._adj:
                self._adj[v] = set()

    def neighbors(self, u: Any) -> Set[Any]:
        return self._adj.get(u, set())

    def bfs(self, source: Any) -> List[Any]:
        seen: Set[Any] = set([source])
        q: Deque[Any] = deque([source])
        order: List[Any] = []
        while q:
            u = q.popleft()
            order.append(u)
            for w in self._adj.get(u, []):
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        return order

    def dfs(self, source: Any) -> List[Any]:
        seen: Set[Any] = set()
        stack: List[Any] = [source]
        order: List[Any] = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            for w in sorted(self._adj.get(u, []), reverse=True):
                if w not in seen:
                    stack.append(w)
        return order

    def shortest_path_unweighted(self, s: Any, t: Any) -> List[Any]:
        if s == t:
            return [s]
        parent: Dict[Any, Any] = {s: None}  # type: ignore[dict-item]
        q: Deque[Any] = deque([s])
        while q:
            u = q.popleft()
            for w in self._adj.get(u, []):
                if w not in parent:
                    parent[w] = u
                    if w == t:
                        path: List[Any] = [t]
                        while parent[path[-1]] is not None:  # type: ignore[index]
                            path.append(parent[path[-1]])     # type: ignore[index]
                        path.reverse()
                        return path
                    q.append(w)
        return []

    def __len__(self) -> int:
        return len(self._adj)


# ---------------------------------------------------------------------
# Demo mínima
# ---------------------------------------------------------------------
def run_demo() -> None:
    print('[demo] Stack/Queue/LinkedList/BST/MinHeap/Graph')

    s = Stack()
    for i in range(3):
        s.push(i)
    print('  stack pop ->', s.pop(), s.pop())

    q = Queue()
    for c in ['a', 'b', 'c']:
        q.enqueue(c)
    print('  queue dequeue ->', q.dequeue(), q.dequeue())

    ll = SinglyLinkedList()
    for v in [10, 20, 30]:
        ll.push_front(v)
    print('  linked list find(20) ->', ll.find(20))
    print('  linked list pop_front ->', ll.pop_front())

    bst = BST()
    for x in [7, 3, 9, 1, 5, 8, 10]:
        bst.insert(x)
    print('  bst search(5) ->', bst.search(5), ' inorder ->', bst.inorder())

    mh = MinHeap([5, 3, 7, 1])
    mh.push(2)
    print('  minheap peek ->', mh.peek(), ' pop ->', mh.pop())

    g = Graph()
    edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)]
    for (u, v) in edges:
        g.add_edge(u, v)
    print('  graph bfs(1) ->', g.bfs(1))
    print('  graph dfs(1) ->', g.dfs(1))
    print('  graph shortest_path 1->6 ->', g.shortest_path_unweighted(1, 6))


# ---------------------------------------------------------------------
# Benchmark simple (inserción/búsqueda BST vs list vs set)
# ---------------------------------------------------------------------
def run_benchmark(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    N = 5000
    data = random.sample(range(10 * N), N)
    probe = random.sample(range(10 * N), 1000)

    lst: List[int] = []
    st: Set[int] = set()
    bst = BST()

    def t_insert_list() -> float:
        t0 = time.perf_counter()
        for x in data:
            lst.append(x)
        return time.perf_counter() - t0

    def t_insert_set() -> float:
        t0 = time.perf_counter()
        for x in data:
            st.add(x)
        return time.perf_counter() - t0

    def t_insert_bst() -> float:
        t0 = time.perf_counter()
        for x in data:
            bst.insert(x)
        return time.perf_counter() - t0

    def t_search_list() -> float:
        t0 = time.perf_counter()
        hits = 0
        for x in probe:
            if x in lst:
                hits += 1
        return time.perf_counter() - t0

    def t_search_set() -> float:
        t0 = time.perf_counter()
        hits = 0
        for x in probe:
            if x in st:
                hits += 1
        return time.perf_counter() - t0

    def t_search_bst() -> float:
        t0 = time.perf_counter()
        hits = 0
        for x in probe:
            if bst.search(x):
                hits += 1
        return time.perf_counter() - t0

    ins_times = [
        ('list.append', t_insert_list()),
        ('set.add', t_insert_set()),
        ('BST.insert', t_insert_bst()),
    ]
    srch_times = [
        ('x in list', t_search_list()),
        ('x in set', t_search_set()),
        ('BST.search', t_search_bst()),
    ]

    csv_path = outdir / 'bench_times.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('op,time_seconds\n')
        for (k, v) in ins_times + srch_times:
            f.write(f'{k},{v:.6f}\n')
    print(f'[bench] tiempos -> {csv_path}')

    try:
        import matplotlib.pyplot as plt  # opcional
        labels = [k for (k, _) in ins_times + srch_times]
        values = [v for (_, v) in ins_times + srch_times]
        idx = list(range(len(values)))

        plt.figure()
        plt.bar(idx, values)
        plt.xticks(idx, labels, rotation=30, ha='right')
        plt.ylabel('segundos (menos es mejor)')
        plt.title(f'Benchmark N={N} (inserción/búsqueda)')
        plt.tight_layout()
        outpng = outdir / 'bench_bar.png'
        plt.savefig(outpng, dpi=140)
        plt.close()
        print(f'[bench] gráfico -> {outpng}')
    except Exception as e:
        print('[bench] no se pudo graficar (matplotlib no disponible?):', e)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Estructuras de datos — demo y benchmark')
    p.add_argument('--demo', action='store_true', help='Corre una demo rápida en consola')
    p.add_argument('--bench', action='store_true', help='Corre benchmark y genera artefactos en ./figs')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo()
    if args.bench:
        run_benchmark(Path(__file__).parent / 'figs')
    if not (args.demo or args.bench):
        print('Nada que hacer. Usa --demo y/o --bench')


if __name__ == '__main__':
    main()
