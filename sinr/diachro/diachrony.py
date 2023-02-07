class DiachronicModels:
    models = list()

    def __init__(self, models):
        self.models = models

    @classmethod
    def mutilayer_factory(cls, graphs):
        return cls()

    @classmethod
    def static_factory(cls, graphs):
        return cls()
