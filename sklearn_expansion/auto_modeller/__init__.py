try:
    from ._autogluon import AutoGluonPredictor

    __all__ = ('AutoGluonPredictor')
except:
    pass