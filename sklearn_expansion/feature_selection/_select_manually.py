
# Editor : Yuki Horie(2021)
# License: BSD 3 clause

from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin

class SelectManually(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """Meta-transformer for selecting features based on importance weights.

    .. versionadded:: 0.17

    Parameters
    ----------
    mask : list of boolean
        The mask of Selector.

    Attributes
    ----------


    Examples
    --------
    >>> from sklearn.feature_selection import SelectFromModel
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform(X)
    array([[-1.34],
           [-0.02],
           [-0.48],
           [ 1.48]])
    """
    def __init__(self, mask):
        self.mask = mask

    def _get_support_mask(self):
        return self.mask

    def fit(self, X, y=None, **fit_params):
        """
        Do Nothing.
        """
        return self

