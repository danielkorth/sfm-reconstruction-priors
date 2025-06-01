#! /bin/bash

python src/precompute_matches.py -m \
    data=v2_30scenes \
    data.num_images=25 \
    sfm/feature_matching=mast3r,sift
