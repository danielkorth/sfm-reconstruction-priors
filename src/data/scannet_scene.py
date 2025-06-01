from pathlib import Path


class ScannetppScene:
    def __init__(
        self, scene_id: str, data_root: str = "/home/korth/guided-research/data/scannetppv2/data/"
    ):
        self.scene_id = Path(scene_id)
        self.data_root = Path(data_root)

    @property
    def scene_dir(self):
        return self.data_root / self.scene_id

    @property
    def dslr_dir(self):
        return self.scene_dir / "dslr"

    @property
    def iphone_dir(self):
        return self.scene_dir / "iphone"

    @property
    def scans_dir(self):
        return self.scene_dir / "scans"

    @property
    def dslr_undistorted_anon_masks_dir(self):
        return self.dslr_dir / "undistorted_anon_masks"

    @property
    def dslr_undistorted_images_dir(self):
        return self.dslr_dir / "undistorted_images"

    @property
    def dslr_nerfstudio_dir(self):
        return self.dslr_dir / "nerfstudio"

    @property
    def dslr_colmap_dir(self):
        return self.dslr_dir / "colmap"

    @property
    def iphone_colmap_dir(self):
        return self.iphone_dir / "colmap"
