class TrainingData:
    def __init__(self):
        self.windows = []
        self.hrm_umls = []
        self.annotations_umls = []
        self.labels = []

    def get(self):
        return {'windows': self.windows,
                'HRM_UMLS': self.hrm_umls,
                'Annotations_UMLS': self.annotations_umls,
                'Labels': self.labels}

    def add_data(self, window, hrm_umls, annotations_umls, label):
        self.windows.append(window)
        self.hrm_umls.append(hrm_umls)
        self.annotations_umls.append(annotations_umls)
        self.labels.append(label)
