class ContextualRelevance:
    def __init__(self, file):
        self.train, self.test = self.init_data(file)

    def init_data(self, file):
        return [], []

    def get_data(self):
        return self.train, self.test
