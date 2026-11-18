INITIAL_STATE = {"condition": "new"}
EXPORT_COLUMN_NAMES = {"age_class": "age_bucket"}
GROUP_NAME = "Condition"
def replacement_state_id(states):
    return 0
def group_labels(states):
    return states["condition"].to_numpy()
