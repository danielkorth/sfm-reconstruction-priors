#! /bin/bash

python src/structure_from_motion.py -m \
    data=v2_30scenes \
    data.num_images=15,20,25 \
    sfm/feature_matching=mast3r \
    cost=ba \
    hparams=mast3r
