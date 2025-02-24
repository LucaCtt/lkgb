import re

from langchain_core.runnables import Runnable

from rov_parser.vector_store import VectorStore


class Parser:
    def __init__(self, chain: Runnable, vector_store: VectorStore) -> "Parser":
        self.chain = chain
        self.vector_store = vector_store

    def compute_template(self, log: str, memory_match_min_quality: int, self_reflection_steps: int) -> None:
        """
        Given a log, this function identifies and updates the template for the log and also for similar logs.
        It first searches for very similar logs in the vector store and checks if their templates match the current log.
        If no matching template is found, it searches for sufficiently similar logs and uses them to generate a template.

        Args:
            log (str): The log for which the template needs to be identified.
            memory_match_min_quality (int): The minimum number of very similar logs required to consider their template.
            self_reflection_steps (int): The number of self-reflection steps to perform to verify the template.

        """
        # Check if there are very similar logs
        # Assumption: the returned documents are sorted by most relevant first
        very_similar_logs = self.vector_store.find_very_similar_logs_with_template(log)

        # If there are very similar logs,
        # check if their template matches with the current log
        if len(very_similar_logs) >= memory_match_min_quality and self.__check_all_templates_match(
            log,
            [similar.metadata["template"] for similar in very_similar_logs],
        ):
            self.vector_store.add_document(log, very_similar_logs[0].metadata["template"])
            return

        # If there are no very similar logs or their template doesn't match,
        # find sufficiently similar logs
        similar_logs = self.vector_store.find_similar_logs(log)

        # Perform self-reflection to verify that the template
        # matches both the current and similar logs
        self_reflection_countdown = self_reflection_steps

        while self_reflection_countdown > 0:
            self_reflection_countdown -= 1

            # Find the template using the current log and the similar logs
            template = self.chain.invoke({"input_log": log, "similar_logs": similar_logs})

            template_regex = self.__template_to_regex(template)

            # Check that the current log matches the template
            if not self.__check_template_match(log, template_regex):
                continue

            # Check that all the similar logs match the template
            for similar_log in similar_logs:
                if not self.__check_template_match(similar_log.page_content, template_regex):
                    continue

            # If the template matches all the logs, stop the self-reflection loop
            break

        # Update the template metadata value for the similar logs
        for similar_log in similar_logs:
            if similar_log.metadata["template"] == template_regex:
                continue

            similar_log.metadata["template"] = template_regex
            self.vector_store.update_document(similar_log)

        # Save the new logs to the vector store
        self.vector_store.add_document(log, template_regex)

    def __template_to_regex(self, template: str) -> str:
        """
        Converts a template string with placeholders into a regular expression string.

        The function replaces the placeholder "<*>" in the template with the regex pattern "(.*?)",
        which matches any character sequence. It also corrects small errors the parser might make.

        Args:
            template (str): The template string containing placeholders.

        Returns:
            str: The resulting regular expression string.

        """
        # Replace <*> with the regex pattern (.*?)
        regex = template.replace("<*>", "(.*?)").strip()

        # Remove any quotes around the regex
        regex = regex.removeprefix("'").removesuffix("'")

        # Remove any extra spaces from the regex
        return regex.strip()

    def __check_template_match(self, log: str, template: str) -> bool:
        """
        Checks if a given log string matches a specified template using regular expressions.

        Args:
            log (str): The log string to be checked.
            template (str): The regular expression template to match against the log string.

        Returns:
            bool: True if the log matches the template, False otherwise. If the template is invalid, returns False.

        """
        try:
            return re.match(template, log) is not None
        except re.error:
            return False

    def __check_all_templates_match(self, log: str, templates: list[str]) -> bool:
        """
        Checks if a given log string matches all the provided regular expression templates.

        Args:
            log (str): The log string to be checked.
            templates (list[str]): A list of regular expression templates to match against the log.

        Returns:
            bool: True if the log matches all the templates, False otherwise. If there is an error in any of the regular expressions, it returns False.

        """
        try:
            return all(re.match(template, log) for template in templates)
        except re.error:
            return False
