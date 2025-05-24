from pathlib import Path
from collections import deque
from typing import Any
from abc import ABC, abstractmethod

import networkx as nx

from practicum_4.dfs import GraphTraversal
from src.plotting.graphs import plot_graph
from src.common import AnyNxGraph


class DfsViaLifoQueueWithPostvisit(GraphTraversal):
    def run(self, node: Any) -> None:
        stack = [node]  #инициализация стека с начальной вершиной
        visited = set()  #для отслеживания посещенных вершин

        while stack:
            current_vertex = stack.pop()  #забор вершины из стека

            if current_vertex not in visited:
                self.previsit(current_vertex)  #вызыв previsit
                visited.add(current_vertex)  #отмечаем вершину как посещенную

                for neighbor in reversed(list(self.graph.neighbors(current_vertex))):
                    if neighbor not in visited:
                        stack.append(neighbor)

                self.postvisit(current_vertex)  #вызываем метод postvisit


class DfsViaLifoQueueWithPrinting(DfsViaLifoQueueWithPostvisit):
    def previsit(self, node: Any, **params) -> None:
        print(f"Previsit node {node}")

    def postvisit(self, node: Any, **params) -> None:
        print(f"Postvisit node {node}")


if __name__ == "__main__":
    #загрузка графа из файла и его обработка
    G = nx.read_edgelist(
        Path("practicum_4") / "simple_graph_10_nodes.edgelist",
        create_using=nx.Graph
    )
    # plot_graph(G)

    dfs = DfsViaLifoQueueWithPrinting(G)
    dfs.run(node="0")  #запуск обхода с начальной вершиной "0"