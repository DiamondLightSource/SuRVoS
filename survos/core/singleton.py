

class Singleton:
    """
    Source: http://stackoverflow.com/a/7346105/764322
    """
    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self, **kwargs):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated(**kwargs)
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)
