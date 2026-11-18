import pandas as pd
CONDITIONS = ["new", "worn", "broken"]
def build_machine_domain_states():
    return pd.DataFrame({"condition": CONDITIONS, "age_class": [0, 1, 2], "state_id": [0, 1, 2]})
