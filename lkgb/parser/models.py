from enum import Enum

from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from pydantic import BaseModel, Field


class EventGraph(BaseModel):
    """Base graph model to represent the information extracted from a log event."""

    nodes: list
    relationships: list


    def graph(self) -> GraphDocument:
        """Convert the event graph to a GraphDocument."""
        nodes_dict = {
            node.id: Node(
                id=node.id,
                type=node.type,
                properties={prop.type: prop.value for prop in node.properties} if node.properties else {},
            )
            for node in self.nodes
        }

        try:
            # The relationships may refer to a node that is not present in the list of nodes.
            # For now, this just fails the parsing process. In the future a number of
            # correction steps could be taken to try to recover from this situation.
            relationships = []
            for rel in self.relationships:
                source_node = nodes_dict[rel.source_id]
                target_node = nodes_dict[rel.target_id]

                relationships.append(Relationship(source=source_node, target=target_node, type=rel.type))
        except KeyError as e:
            msg = "Relationship refers to a node that is not present in the list of nodes"
            raise ValueError(msg) from e

        return GraphDocument(nodes=list(nodes_dict.values()), relationships=relationships)


def build_dynamic_model(ontology: GraphDocument) -> type[EventGraph]:
    """Build a dynamic event graph model based on the ontology."""
    valid_node_types = [node.type for node in ontology.nodes]
    valid_relationships = [rel.type for rel in ontology.relationships]
    valid_triples = [(rel.source.type, rel.type, rel.target.type) for rel in ontology.relationships]

    valid_properties: list[str] = ["uri"]
    valid_properties_dict: dict[str, list[str]] = {}
    for node in ontology.nodes:
        valid_properties = valid_properties + list(node.properties.keys())
        valid_properties_dict[node.type] = list(node.properties.keys())

    valid_properties_schema = [f"{node}:{props}" for node, props in valid_properties_dict.items()]

    _NodeType = Enum("_NodeType", {node: node for node in valid_node_types}, type=str)  # noqa: N806
    _PropertyType = Enum("_PropertyType", {prop: prop for prop in valid_properties}, type=str)  # noqa: N806
    _RelationshipType = Enum("_RelationshipType", {rel: rel for rel in valid_relationships}, type=str)  # noqa: N806

    class _Property(BaseModel):
        type: _PropertyType = Field(  # type: ignore[valid-type]
            description=f"The type or label of the property. Available options are: {valid_properties}",
        )
        value: str | float = Field(description=("Extracted value."))

    class _Node(BaseModel):
        id: str = Field(description="Name or human-readable unique identifier.")
        type: _NodeType = Field(  # type: ignore[valid-type]
            description=f"The type or label of the node. Available options are: {valid_node_types}",
        )
        properties: list[_Property] | None = Field(default=None, description="List of node properties.")

    # The doc is set this way to for string interpolation
    _Node.__doc__ = (
        "Each node type has a specific set of available properties. "
        f"The available properties for each node type are: {valid_properties_schema} "
    )

    class _Relationship(BaseModel):
        source_id: str = Field(description="Name or human-readable unique identifier of source node.")
        target_id: str = Field(description="Name or human-readable unique identifier of source node.")
        type: _RelationshipType = Field(  # type: ignore[valid-type]
            description=f"The type or label of the relationship. Available types are: {valid_relationships}",
        )

    _Relationship.__doc__ = (
        "Each relationship type has a specific source and target node type. "
        "The available relationships specified in the format "
        f"(source type, relationship type, target type) are: {valid_triples} "
    )

    class DynamicEventGraph(EventGraph):
        nodes: list[_Node] = Field(description="List of nodes.")
        relationships: list[_Relationship] = Field(
            description="List of relationships.",
        )

    return DynamicEventGraph
