from datetime import UTC, datetime


class ParserReport:
    """A class to generate and manage reports for parsing operations.

    All of the dates and times are in UTC.
    """

    def __init__(
        self,
    ) -> "ParserReport":
        self.start_dt = datetime.now(tz=UTC)

    def finish(self) -> "ParserReport":
        """Mark the end of the parsing process by setting the end datetime.

        Returns:
            ParserReport: The instance of the ParserReport with the updated end datetime.

        """
        self.end_dt = datetime.now(tz=UTC)
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

    def avg_total_time_taken(self) -> float:
        """Calculate the average total time taken from all parser reports.

        Returns:
            float: The average total time taken.

        """
        return sum(report.total_time_taken() for report in self.parser_reports) / len(self.parser_reports)
