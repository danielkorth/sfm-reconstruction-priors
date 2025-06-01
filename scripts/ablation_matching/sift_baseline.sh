#! /bin/bash

python src/structure_from_motion.py -m \
    data=v2_10scenes_2 \
    cost=ba \
    sfm/feature_matching=sift \
    data.num_images=15
