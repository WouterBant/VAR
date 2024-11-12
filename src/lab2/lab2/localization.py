class Localization:
    def __init__(self, config):
        self.config = config
        self.marker_positions = {
            0: (0.0, 0.0),
            1: (0.0, 1.0),
            2: (1.0, 1.0),
        }
        self.marker_sizes = {
            0: 0.1,
            1: 0.1,
            2: 0.1,
        }

    def triangulate(self, marker_distances):
        pass
