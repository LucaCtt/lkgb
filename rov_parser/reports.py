from datetime import UTC, datetime


class ParserReport:
    """A class to generate and manage reports for parsing operations.

    All of the dates and times are in UTC.
    """

    def __init__(
        self,
    ) -> "ParserReport":
        self.start_dt = datetime.now(tz=UTC)

    def find_very_similar_logs_done(self) -> None:
        """Set the timestamp indicating when very similar logs were last processed.

        Returns:
            None

        """
        self.very_similar_logs_dt = datetime.now(tz=UTC)

    def find_similar_logs_done(self) -> None:
        """Set the timestamp for when similar logs were last found.

        Returns:
            None

        """
        self.similar_logs_dt = datetime.now(tz=UTC)

    def template_generation_done(self) -> None:
        """Mark the template generation as completed by setting the timestamp.

        Returns:
            None

        """
        self.template_generation_dt = datetime.now(tz=UTC)

    def finish(self) -> "ParserReport":
        """Mark the end of the parsing process by setting the end datetime.

        Returns:
            ParserReport: The instance of the ParserReport with the updated end datetime.

        """
        self.end_dt = datetime.now(tz=UTC)
        return self

    def is_memory_match(self) -> bool:
        """Check if the parsing ended with a memory match.

        Returns:
            bool: True if `end_dt` is not None and `similar_logs_dt` is None, False otherwise.

        """
        return self.end_dt is not None and self.similar_logs_dt is None

    def very_similar_logs_time_taken(self) -> float:
        """Calculate the time taken between the start time and the time of very similar logs.

        Returns:
            float: The time difference in seconds.

        """
        return (self.very_similar_logs_dt - self.start_dt).total_seconds()

    def similar_logs_time_taken(self) -> float:
        """Calculate the time difference in seconds between similar logs and very similar logs.

        Returns:
            float: The time difference in seconds.

        """
        return (self.similar_logs_dt - self.very_similar_logs_dt).total_seconds()

    def template_generation_time_taken(self) -> float:
        """Calculate the time taken for template generation.

        This method computes the difference in time between the template generation
        datetime and the similar logs datetime, and returns the result in seconds.

        Returns:
            float: The time taken for template generation in seconds.

        """
        return (self.template_generation_dt - self.similar_logs_dt).total_seconds()

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

    def percentage_memory_matches(self) -> float:
        """Calculate the percentage of reports that are memory matches.

        This method iterates through the list of parser reports and counts how many of them
        are memory matches. It then divides this count by the total number of reports to
        get the percentage of memory matches.

        Returns:
            float: The percentage of reports that are memory matches.

        """
        return sum(1 for report in self.parser_reports if report.is_memory_match()) / len(self.parser_reports)

    def avg_very_similar_logs_time_taken(self) -> float:
        """Calculate the average time taken for very similar logs across all parser reports.

        Returns:
            float: The average time taken for very similar logs.

        """
        return sum(report.very_similar_logs_time_taken() for report in self.parser_reports) / len(self.parser_reports)

    def avg_similar_logs_time_taken(self) -> float:
        """Calculate the average time taken for similar logs across all parser reports.

        This method filters out None values from the list of times taken for similar logs
        and computes the average of the remaining times.

        Returns:
            float: The average time taken for similar logs. If there are no valid times,
               the function may raise a ZeroDivisionError.

        """
        times_without_nones = filter(None, [report.similar_logs_time_taken() for report in self.parser_reports])
        return sum(times_without_nones) / len(self.parser_reports)

    def avg_template_generation_time_taken(self) -> float:
        """Calculate the average time taken for template generation across all parser reports.

        This method filters out any None values from the template generation times of the reports
        and computes the average time taken.

        Returns:
            float: The average template generation time taken.

        """
        times_without_nones = filter(None, [report.template_generation_time_taken() for report in self.parser_reports])
        return sum(times_without_nones) / len(self.parser_reports)

    def avg_total_time_taken(self) -> float:
        """Calculate the average total time taken from all parser reports.

        Returns:
            float: The average total time taken.

        """
        return sum(report.total_time_taken() for report in self.parser_reports) / len(self.parser_reports)
