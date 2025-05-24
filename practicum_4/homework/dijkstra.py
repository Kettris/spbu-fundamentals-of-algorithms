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
        self.shortest_paths: dict[Any, list[Any]] = {} #хранит кратчайшие пути
        super().__init__(G)

    def previsit(self, node: Any, **params) -> None:
        self.shortest_paths[node] = params["path"] #кратч путь к shortest_paths

    def postvisit(self, node: Any, **params) -> None:
        pass

    def run(self, start_vertex: Any) -> dict[Any, float]:
        distances = {vertex: float('inf') for vertex in self.graph} #хранит растояние до всех вершин
        distances[start_vertex] = 0
        priority_queue = [(0, start_vertex)]
        self.shortest_paths[start_vertex] = [start_vertex]

        while priority_queue: 
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]: #узел с наим эл
                continue 
            for neighbor in self.graph[current_vertex]: #если при проверке вершина устарела перезаписыв
                weight = self.graph[current_vertex][neighbor].get('weight', 1) #по соседям
                distance = current_distance + weight #растояние до соседа

                if distance < distances[neighbor]: #проверк нов растояния что кротчайший
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
                    self.shortest_paths[neighbor] = self.shortest_paths[current_vertex] + [neighbor]

        return distances #обнов кратч путей

if __name__ == "__main__":
    G = nx.read_edgelist(
        Path("practicum_4") / "simple_weighted_graph_9_nodes.edgelist",
        create_using=nx.Graph
    )
    plot_graph(G)

    sp = DijkstraAlgorithm(G)
    sp.run("0")

    test_node = "5"
    shortest_path_edges = [
        (sp.shortest_paths[test_node][i], sp.shortest_paths[test_node][i + 1])
        for i in range(len(sp.shortest_paths[test_node]) - 1)
    ]
    plot_graph(G, highlighted_edges=shortest_path_edges)
