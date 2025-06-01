import numpy as np


class Matcher:
    """A matcher takes descriptors from 2 different images and matches them together into :Match:

    objects.
    """

    def __init__(self):
        pass

    def match(self, query_des, database_des, topk=1, distance_metric="cosine", max_distance=0.7):
        """Find topk closest matches for each query descriptor comparing it to all train
        descriptors.

        Args:
            query_des (_type_): query descriptors (from a query image)
            database_des (_type_): database descriptors (from a database) - either single descriptor or list of descriptors
            topk (int, optional): _description_. Defaults to 2.
            max_distance (float, optional): maximum distance for a match to be considered a good match. Defaults to 0.7.

        Returns:
            _type_: _description_
        """
        matches = []

        # if multiple database images
        if isinstance(database_des, list):
            num_des_per_img = [des.shape[0] for des in database_des]
            database_des = np.concatenate(database_des, axis=0)
        # if single database image
        elif len(database_des.shape) == 2:
            num_des_per_img = [database_des.shape[0]]

        for i, des in enumerate(query_des):
            if distance_metric == "euclidean":
                # based on euclidean distance
                distances = np.linalg.norm(des - database_des, axis=1)
                indices = np.argpartition(distances, range(topk))[:topk]

                if distances[indices[0]] > max_distance:
                    continue

            elif distance_metric == "cosine":
                # https://github.com/colmap/colmap/blob/cf8f116197ff0e2ac869d04d3f41792ca18da449/src/colmap/feature/sift.cc#L734
                # https://stackoverflow.com/questions/55962820/what-is-the-problem-of-matching-sift-descriptor-by-using-euclidean-distance
                cos_theta = np.dot(
                    des, database_des.T
                )  # / (np.linalg.norm(des) * np.linalg.norm(database_des, axis=1))
                indices = np.argpartition(cos_theta, -topk)[-topk:][::-1]
                distances = np.arccos(np.minimum(cos_theta, 1))

                if distances[indices[0]] > max_distance:
                    continue

            matches_per_des = []

            for idx in indices:
                # find database image index
                img_idx = 0
                per_img_idx = idx
                for num in num_des_per_img:
                    if per_img_idx < num:
                        break
                    per_img_idx -= num
                    img_idx += 1
                matches_per_des.append(
                    Match(
                        query_idx=i,
                        database_idx=per_img_idx,
                        distance=distances[idx],
                        database_img=img_idx,
                    )
                )

            matches.append(matches_per_des)

        return matches


class Match:
    """A Match object represents a match between a query descriptor and a train descriptor.

    This object is usually considered one
    """

    def __init__(self, query_idx, database_idx, distance=None, database_img=None):
        self.query_idx = query_idx
        self.database_idx = database_idx
        self.distance = distance
        self.database_img = database_img

    # utility function for debugging
    def printvals(self):
        return f"Match: query_idx={self.query_idx}, database_idx={self.database_idx}, distance={self.distance}, database_img={self.database_img}"

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__qualname__}: ({self.query_idx}, {self.database_idx})>"
