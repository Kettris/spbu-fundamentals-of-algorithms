from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx
import heapq

from practicum_4.dfs import GraphTraversal
from src.plotting.graphs import plot_graph
from src.common import AnyNxGraph


class DijkstraAlgorithm(GraphTraversal):
    def __init__(self, G: AnyNxGraph) -> None:
        self.shortest_paths: dict[Any, list[Any]] = {}
        super().__init__(G)

    def previsit(self, node: Any, **params) -> None:
        self.shortest_paths[node] = params["path"]

    def postvisit(self, node: Any, **params) -> None:
        pass

    def run(self, start_vertex: Any) -> None:
        distances = {vertex: float('inf') for vertex in self.graph}
        distances[start_vertex] = 0
        priority_queue = [(0, start_vertex)]  # расстояние, вершина
        self.shortest_paths[start_vertex] = [start_vertex]  #путь до стартовой вершины

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)  #вершина с мин расстоянием

            if current_distance > distances[current_vertex]:
                continue  #пропускаем, если расстояние не то

            #проход по всем соседям текущей вершины
            for neighbor, weight in self.graph[current_vertex].items():
                distance = current_distance + weight

                #если есть меньшее расстояние к соседу, обновляем
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
                    self.shortest_paths[neighbor] = self.shortest_paths[current_vertex] + [neighbor]  # Сохраняем путь

        return distances


if __name__ == "__main__":
    #загрузка графа
    G = nx.read_edgelist(
        Path("practicum_4") / "simple_weighted_graph_9_nodes.edgelist",
        create_using=nx.Graph
    )
    plot_graph(G)

    dijkstra = DijkstraAlgorithm(G)
    dijkstra.run("0")  #запуск алгоритма Дейкстры с начальной вершиной "0"

    test_node = "5"
    shortest_path_edges = [
        (dijkstra.shortest_paths[test_node][i], dijkstra.shortest_paths[test_node][i + 1])
        for i in range(len(dijkstra.shortest_paths[test_node]) - 1)
    ]
    plot_graph(G, highlighted_edges=shortest_path_edges)