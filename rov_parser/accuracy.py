import pandas as pd


def evaluate_result(pred_file: str, true_file: str) -> tuple[float, float]:
    df_true = pd.read_csv(true_file, usecols=["Content", "EventId", "EventTemplate"])

    df_true["template_no_spaces"] = (
        df_true["EventTemplate"].str.replace(r"\s+", "", regex=True).str.replace(r"\<\*\>", "", regex=True)
    )

    df_pred = pd.read_csv(pred_file, index_col=False, usecols=["text", "template"])

    df_pred["template_no_spaces"] = (
        df_pred["template"].str.replace(r"\s+", "", regex=True).str.replace(r"\(\.\*\?\)", "", regex=True)
    )

    group_acc = get_group_accuracy(df_true["template_no_spaces"], df_pred["template_no_spaces"])

    correctly_parsed_messages = df_pred["template_no_spaces"].eq(df_true["template_no_spaces"]).to_numpy().sum()
    parsing_acc = float(correctly_parsed_messages) / len(df_pred[["Content"]])

    return group_acc, parsing_acc


def get_group_accuracy(templates_true: pd.Series, templates_pred: pd.Series) -> float:
    templated_pred_valuecounts = templates_pred.value_counts()

    accurate_events_count = 0
    for parsed_event_id in templated_pred_valuecounts.index:
        log_ids = templates_pred[templates_pred == parsed_event_id].index
        templates_true_log_id_valuecounts = templates_true[log_ids].value_counts()
        if templates_true_log_id_valuecounts.size == 1:
            true_event_id = templates_true_log_id_valuecounts.index[0]
            if log_ids.size == templates_true[templates_true == true_event_id].size:
                accurate_events_count += log_ids.size

    return float(accurate_events_count) / templates_true.size
