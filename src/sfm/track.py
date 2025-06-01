from collections import defaultdict
from enum import Enum
from typing import List

import numpy as np

from sfm.view import View


class TrackState(Enum):
    INACTIVE = 0
    OBSERVED = 1
    ACTIVE = 2
    REMOVED = 3


class Track:
    """A track is a set of features in different images that correspond to the same 3D location.

    A track can be in different states:
    - inactive: the track has not been observed by any view
    - observed: the track can be observed by at least one view that is active in the reconstruction
    - active: the track is an active part of the 3d reconstruction
    - removed: the track was once active, but has been removed due to geometric constraints
    """

    def __init__(self):
        self._point = None
        self.views = list()
        self.views_observing = set()
        self._state = TrackState.INACTIVE

    def add_view(self, view: int, keypoint_idx: int):
        """Add a view that observes the current track :view: the view object that is added to the
        track :keypoint_idx: the keypoint_idx of the keypoint that corresponds to the track."""
        self.views.append((view, keypoint_idx))

    @property
    def point(self):
        if self._point is None:
            raise ValueError("Track has not been initialized yet.")
        return self._point

    @point.setter
    def point(self, value):
        self._point = value

    def check_consistent(self):
        """Check if the track is consistent or if it contains multiple locations on the same image.

        the track is consistent if it contains at most one feature of every view!
        """
        # remove if it was added multiple times
        views = [a[0] for a in self.views]
        if len(views) > len(set(views)):
            return False
            # not implemented becuaes it should not happen as of now
            # raise ValueError("Track is not consistent!")
        return True

    def set_view_active(self, view_idx):
        self.views_observing.add(view_idx)

    @property
    def active_views(self):
        return self.views_observing

    @property
    def inactive_views(self):
        return [view[0] for view in self.views if view[0] not in self.views_observing]

    def find_feature_for_view(self, view):
        for view_pair in self.views:
            if view_pair[0] == view:
                feature = view_pair[1]
                return feature
        raise ValueError("View is not part of the Track")

    def contains(self, view):
        for view_pair in self.views:
            if view_pair[0] == view:
                return True
        return False

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: TrackState):
        if self._state == TrackState.REMOVED:
            raise ValueError("Track is removed and cannot be set to a different state")
        self._state = state
