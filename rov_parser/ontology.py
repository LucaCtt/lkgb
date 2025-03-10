from rdflib import Graph


def _get_local_name(uri: str) -> str:
    if isinstance(uri, str) and ("/" in uri or "#" in uri):
        return uri.split("/")[-1].split("#")[-1]  # Handles both "/" and "#" separators
    return uri


class LogOntology:
    def __init__(self, path: str, rdf_format: str = "turtle") -> "LogOntology":
        self.g = Graph()
        self.g.parse(path, format=rdf_format)

    def classes(self) -> list[tuple[str, str]]:
        """
        Retrieve a list of ontology classes with their descriptions.

        This method executes a SPARQL query to fetch all distinct classes
        (both `owl:Class` and `rdfs:Class`) from the ontology graph. It also
        attempts to retrieve optional descriptions for each class.

        Returns:
            A list of tuples containing the class name and its description.

        """
        query = """
            SELECT DISTINCT ?class ?description WHERE {
                { ?class a owl:Class . }
                UNION
                { ?class a rdfs:Class . }
                OPTIONAL { ?class rdfs:comment ?description . }
            }
        """

        classes: list[tuple[str, str]] = []
        for row in self.g.query(query):
            class_name = _get_local_name(str(row[0]))
            description = str(row[1]) if row[1] else "No description available."
            description = description.replace("\n", " ")
            if description[-1:] != ".":
                description += "."
            classes.append((class_name, description))

        return classes
