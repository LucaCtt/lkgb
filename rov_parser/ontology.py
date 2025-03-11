from rdflib import Graph


def _get_local_name(uri: str) -> str:
    if isinstance(uri, str) and ("/" in uri or "#" in uri):
        return uri.split("/")[-1].split("#")[-1]  # Handles both "/" and "#" separators
    return uri


class SlogertOntology:
    event_class = "Event"

    def __init__(self, path: str, rdf_format: str = "turtle") -> "SlogertOntology":
        self.g = Graph()
        self.g.parse(path, format=rdf_format)

    def classes(self) -> list[tuple[str, str]]:
        """
        Retrieve a list of (non-event) ontology classes with their descriptions.

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
            FILTER (?class != <https://w3id.org/sepses/ns/log#Event>)
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

    def event_properties(self) -> list[tuple[str, str, str]]:
        """
        Retrieve a list of ontology properties with their descriptions and domains.

        This method executes a SPARQL query to fetch all distinct properties
        (both `owl:ObjectProperty` and `rdf:Property`) from the ontology graph.
        It also attempts to retrieve optional descriptions and domains for each property.

        Returns:
            A list of tuples containing the property name, its description, and its domain.

        """
        props_query = """
            SELECT DISTINCT ?property ?range ?description WHERE {
            { ?property a owl:ObjectProperty . }
            UNION
            { ?property a rdf:Property . }
            ?property rdfs:range ?range .
            OPTIONAL { ?property rdfs:comment ?description . }
            OPTIONAL { ?property rdfs:domain ?domain . }
            FILTER (?domain = <https://w3id.org/sepses/ns/log#Event>)
            }
        """

        sub_props_query = """
            SELECT DISTINCT ?subProperty ?range ?description WHERE {
            ?subProperty rdfs:subPropertyOf ?property .
            ?property rdfs:range ?range .
            OPTIONAL { ?subProperty rdfs:comment ?description . }
            }
        """

        props: list[tuple[str, str, str]] = []
        for row in self.g.query(props_query):
            prop_name = _get_local_name(str(row[0]))
            prop_range = _get_local_name(str(row[1]))

            desc = str(row[2]) if row[2] else "No description available."
            desc = desc.replace("\n", " ")
            if desc[-1:] != ".":
                desc += "."

            props.append((prop_name, prop_range, desc))

            sub_props_rows = self.g.query(sub_props_query, initBindings={"property": row[0]})
            for sub_row in sub_props_rows:
                sub_prop_name = _get_local_name(str(sub_row[0]))
                sub_prop_range = _get_local_name(str(row[1]))

                sub_desc = str(sub_row[2]) if sub_row[2] else "No description available."
                sub_desc = sub_desc.replace("\n", " ")
                if sub_desc[-1:] != ".":
                    sub_desc += "."

                props.append((sub_prop_name, sub_prop_range, sub_desc))

        return props

    def get_event_object_property(self, obj_range: str) -> str | None:
        """
        Retrieve an event object property based on the provided range.

        This method executes a SPARQL query to fetch an object property
        that has the provided range as its domain.

        Args:
            obj_range (str): The range of the object property.

        Returns:
            The name of the object property.

        """
        query = f"""
            SELECT DISTINCT ?property WHERE {{
            ?property a owl:ObjectProperty .
            ?property rdfs:range <https://w3id.org/sepses/ns/log#{obj_range}> .
            ?property rdfs:domain <https://w3id.org/sepses/ns/log#Event> .
            }}
            ORDER BY DESC(?property)
            LIMIT 1
        """

        for row in self.g.query(query):
            return _get_local_name(str(row[0]))

        return None
