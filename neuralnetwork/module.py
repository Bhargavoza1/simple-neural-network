# Base class
class Module:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, input):
        raise NotImplementedError