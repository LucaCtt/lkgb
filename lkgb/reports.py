"""Classes to generate and manage reports for parsing operations."""

from datetime import UTC, datetime

from langchain_neo4j.graphs.graph_document import GraphDocument


class ParserReport:
    """A class to generate and manage reports for parsing operations.

    All of the dates and times are in UTC.
    """

    def __init__(
        self,
    ) -> "ParserReport":
        self.start_dt = datetime.now(tz=UTC)
        self.error: Exception | str | None = None
        self.graph: GraphDocument | None = None

    def failure(self, error: Exception) -> "ParserReport":
        """Mark the end of the parsing process by setting the end datetime.

        Args:
            error (Exception): The error that occurred during the parsing process.

        Returns:
            ParserReport: The instance of the ParserReport with the updated end datetime.

        """
        self.end_dt = datetime.now(tz=UTC)
        self.error = error

        return self

    def success(self, graph: GraphDocument) -> "ParserReport":
        """Mark the end of the parsing process by setting the end datetime.

        Args:
            graph (GraphDocument): The graph document created by the parser.

        Returns:
            ParserReport: The instance of the ParserReport with the updated end datetime.

        """
        self.end_dt = datetime.now(tz=UTC)
        self.graph = graph

        return self

    def total_time_taken(self) -> float:
        """Calculate the total time taken for an event.

        This method computes the difference between the end time and the start time
        of an event and returns the total duration in seconds.

        Returns:
            float: The total time taken in seconds.

        """
        return (self.end_dt - self.start_dt).total_seconds()


class RunSummary:
    """A class to summarize the results of multiple parser reports."""

    def __init__(self, parser_reports: list[ParserReport]) -> "RunSummary":
        self.parser_reports = parser_reports

    def parse_time_average(self) -> float:
        """Calculate the average total time taken from all parser reports.

        Returns:
            float: The average total time taken.

        """
        return sum(report.total_time_taken() for report in self.parser_reports) / len(self.parser_reports)

    def success_percentage(self) -> float:
        """Calculate the percentage of successful parser reports.

        Returns:
            float: The percentage of successful parser reports.

        """
        return len([report for report in self.parser_reports if not report.error]) / len(self.parser_reports)
