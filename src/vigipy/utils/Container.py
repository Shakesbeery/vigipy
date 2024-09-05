import pandas as pd


class Container:
    def __init__(self, params=False):
        if params:
            self.param = dict()

    def export(self, name, index=False):
        with pd.ExcelWriter(name) as writer:
            self.signals.to_excel(writer, sheet_name="Signals", index=index)
            self.all_signals.to_excel(writer, sheet_name="all_data", index=index)
