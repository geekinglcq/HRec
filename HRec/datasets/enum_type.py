from enum import Enum


class FeatureSource(Enum):
    """Source of features.

    - ``INTERACTION``: Features from ``.inter``
    - ``USER``: Features from ``.user`` (other than ``user_id``).
    - ``ITEM``: Features from ``.item`` (other than ``item_id``).
    - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
    - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
    """

    INTERACTION = 'inter'
    USER = 'user'
    ITEM = 'item'
    USER_ID = 'user_id'
    ITEM_ID = 'item_id'


item_type_dict = {'book': 0.0, 'music': 1.0, 'movie': 2.0}
