from pathlib import Path
from collections import deque
from typing import Any
from abc import ABC, abstractmethod

import networkx as nx

from practicum_4.dfs import GraphTraversal
from src.plotting.graphs import plot_graph
from src.common import AnyNxGraph


class DfsViaLifoQueueWithPostvisit(GraphTraversal): # обход графа
    def run(self, node: Any) -> None:
        stack = [(node, False)]
        visited = set()

        while stack:
            current_vertex, processed = stack.pop()
            if processed:
                self.postvisit(current_vertex)
                continue
            if current_vertex in visited:
                continue
                
            self.previsit(current_vertex)
            
            visited.add(current_vertex)
            stack.append((current_vertex, True))
            
            for neighbor in reversed(list(self.graph.neighbors(current_vertex))):
                if neighbor not in visited:
                    stack.append((neighbor, False))


class DfsViaLifoQueueWithPrinting(DfsViaLifoQueueWithPostvisit):
    def previsit(self, node: Any, **params) -> None:
        print(f"Previsit node {node}")

    def postvisit(self, node: Any, **params) -> None:
        print(f"Postvisit node {node}")


if __name__ == "__main__":
    G = nx.read_edgelist(
        Path("practicum_4") / "simple_graph_10_nodes.edgelist",
        create_using=nx.Graph
    )
    # plot_graph(G)

    dfs = DfsViaLifoQueueWithPrinting(G)
    dfs.run(node="0")
