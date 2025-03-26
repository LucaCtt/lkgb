"""Parser accuracy metrics."""

import networkx as nx
from langchain_neo4j.graphs.graph_document import GraphDocument


def __graph_document_to_networkx(graph: GraphDocument) -> nx.Graph:
    g = nx.Graph()
    for node in graph.nodes:
        g.add_node(node.id, **node.properties)

    for relationship in graph.relationships:
        g.add_edge(relationship.source.id, relationship.target.id, label=relationship.type)
    return g


def graph_edit_distance(graph1: GraphDocument, graph2: GraphDocument) -> float:
    gn1 = __graph_document_to_networkx(graph1)
    gn2 = __graph_document_to_networkx(graph2)

    return nx.graph_edit_distance(gn1, gn2)
