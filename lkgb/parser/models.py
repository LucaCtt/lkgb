from enum import Enum

from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from pydantic import BaseModel, Field, model_validator


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

        relationships = [
            Relationship(source=nodes_dict[rel.source_id], target=nodes_dict[rel.target_id], type=rel.type)
            for rel in self.relationships
        ]

        return GraphDocument(nodes=list(nodes_dict.values()), relationships=relationships)


def build_dynamic_model(ontology: GraphDocument) -> type[EventGraph]:
    """Build a dynamic event graph model based on the ontology."""
    valid_node_types = [node.type for node in ontology.nodes]
    valid_properties_per_node = {node.type: [*list(node.properties.keys()), "uri"] for node in ontology.nodes}
    valid_properties = list({prop for props in valid_properties_per_node.values() for prop in props})

    valid_relationship_types = [rel.type for rel in ontology.relationships]
    valid_triples = [(rel.source.type, rel.type, rel.target.type) for rel in ontology.relationships]

    valid_properties_schema = [f"{node}:{props}" for node, props in valid_properties_per_node.items()]

    NodeType = Enum("NodeType", {node: node for node in valid_node_types}, type=str)  # noqa: N806
    PropertyType = Enum("PropertyType", {prop: prop for prop in valid_properties}, type=str)  # noqa: N806
    RelationshipType = Enum("RelationshipType", {rel: rel for rel in valid_relationship_types}, type=str)  # noqa: N806

    class Property(BaseModel):
        """A property of a node in the event graph."""

        type: PropertyType = Field(  # type: ignore[valid-type]
            description=f"Type of the property. Must be one of: {valid_properties}",
        )
        value: str | int | float = Field(description="Extracted value of the property.")

    class Node(BaseModel):
        id: str = Field(description="Unique identifier for the node.")
        type: NodeType = Field(  # type: ignore[valid-type]
            description=f"Type of the node. Must be one of: {valid_node_types}",
        )
        properties: list[Property] | None = Field(default=None, description="List of properties of the node.")

        __doc__ = (
            "A node in the event graph. "
            "Each node type has a specific set of allowed properties. "
            f"The allowed properties for each node type are: {valid_properties_schema} "
        )

    class Relationship(BaseModel):
        source_id: str = Field(description="Unique identifier of source node.")
        target_id: str = Field(description="Unique identifier of target node.")
        type: RelationshipType = Field(  # type: ignore[valid-type]
            description=f"Type of the relationship. Must be one of: {valid_relationship_types}",
        )

        __doc__ = (
            "A relationship between two nodes in the event graph. "
            "Each relationship type has a predefined source and target node type. "
            "The allowed relationships, formatted as (source type, relationship type, target type), are: "
            f"{valid_triples}."
        )

    class DynamicEventGraph(EventGraph):
        """Represents a dynamic event-based knowledge graph composed of nodes and relationships."""

        nodes: list[Node] = Field(description="List of nodes in the graph.")
        relationships: list[Relationship] = Field(
            description="List of relationships in the graph.",
        )

        @model_validator(mode="after")
        def validate_relationships(self) -> "DynamicEventGraph":
            node_ids = {node.id for node in self.nodes}

            missing_nodes = {rel.source_id for rel in self.relationships} | {
                rel.target_id for rel in self.relationships
            } - node_ids

            if missing_nodes:
                msg = f"Relationships mention node ids that are not present in the nodes list: {missing_nodes}"
                raise ValueError(msg)

            return self

    return DynamicEventGraph
