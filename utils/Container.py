import pandas as pd

class Container():

    def __init__(self, params=False):
        if params:
            self.param = dict()

    def export(self, name, index=False):
        writer = pd.ExcelWriter(name)
        self.signals.to_excel(writer,'Signals', index=index)
        self.all_signals.to_excel(writer,'all_data', index=index)
        writer.save()