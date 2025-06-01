#! /bin/bash

python src/sfm_colmap.py -m \
    data=v2_30scenes \
    data.num_images=15,20,25
