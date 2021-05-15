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

    def add_data(self, window, hrm_umls, annotations_umls, label, is_expanded=0):
        self.windows.append(window)
        self.hrm_umls.append(hrm_umls)
        self.annotations_umls.append(annotations_umls)
        self.labels.append(label)


class TrainingDataExtended(TrainingData):
    def __init__(self):
        TrainingData.__init__(self)
        self.is_expanded_term = []

    def get(self):
        data = TrainingData.get(self)
        data['Is_Expanded_Term'] = self.is_expanded_term
        return data

    def add_data(self, window, hrm_umls, annotations_umls, label, is_expanded=0):
        TrainingData.add_data(self, window, hrm_umls, annotations_umls, label)
        self.is_expanded_term.append(is_expanded)
