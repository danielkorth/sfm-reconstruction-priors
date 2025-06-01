from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from jaxtyping import Float


class Pointmap(ABC):
    """Abstract base class for pointmaps.

    A pointmap represents 3D points in a consistent coordinate frame. This can be either a two-view
    pointmap (from two images) or a multiview pointmap (from multiple images).
    """

    def __init__(
        self,
        pts3d: Float[torch.Tensor, "N H W 3"],
        conf: Optional[Float[torch.Tensor, "N H W"]] = None,
    ):
        """Initialize the pointmap.

        Args:
            resolution: The resolution of the pointmap (height, width)
        """
        self._pts3d = pts3d
        self._conf = conf
        self.height, self.width = pts3d.shape[1:3]

    @property
    def pts3d(self) -> torch.Tensor:
        """Get the 3D points tensor.

        Returns:
            3D points tensor of shape N×H×W×3
        """
        return self._pts3d

    @property
    def conf(self) -> torch.Tensor:
        """Get the confidence tensor.

        Returns:
            Confidence tensor of shape N×H×W
        """
        return self._conf

    def get_point_at(
        self,
        x: Union[float, torch.Tensor],
        y: Union[float, torch.Tensor],
        view_idx: Optional[Union[int, torch.Tensor]] = None,
        use_interpolation: bool = False,
    ) -> torch.Tensor:
        """Get the 3D point at the given 2D coordinates.

        Args:
            x: X coordinate(s) (column) - can be a single float or a torch.Tensor
            y: Y coordinate(s) (row) - can be a single float or a torch.Tensor
            view_idx: Index of the view to get the point from (if None, returns points from all views)
                      Can be a single int, a torch.Tensor, or None
            use_interpolation: Whether to use bilinear interpolation

        Returns:
            3D point(s) as torch tensor
        """
        # Convert single values to tensors for consistent handling
        if isinstance(x, (int, float)):
            x = torch.tensor([x])
        if isinstance(y, (int, float)):
            y = torch.tensor([y])

        # Ensure x and y have the same shape
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        # Handle view indices
        if view_idx is not None:
            if isinstance(view_idx, int):
                view_idx = torch.tensor([view_idx])

            # Ensure view indices are within bounds
            if torch.any(view_idx < 0) or torch.any(view_idx >= self.pts3d.shape[0]):
                raise ValueError(f"View indices must be in range [0, {self.pts3d.shape[0]-1}]")

            # If x and y are single-element tensors and view_idx is a single-element tensor,
            # we can directly index the pointmap
            if x.numel() == 1 and y.numel() == 1 and view_idx.numel() == 1:
                pointmap = self.pts3d[view_idx.item()]

                if use_interpolation:
                    return self.bilinear_interpolate_3d(pointmap, x.item(), y.item())
                else:
                    x_int, y_int = int(round(x.item())), int(round(y.item()))
                    # Ensure coordinates are within bounds
                    x_int = max(0, min(x_int, self.width - 1))
                    y_int = max(0, min(y_int, self.height - 1))
                    return pointmap[y_int, x_int]

            # For multiple points, we need to handle each one
            points = []
            for i in range(len(x)):
                idx = view_idx[i].item() if i < len(view_idx) else i
                pointmap = self.pts3d[idx]

                if use_interpolation:
                    points.append(self.bilinear_interpolate_3d(pointmap, x[i].item(), y[i].item()))
                else:
                    x_int, y_int = int(round(x[i].item())), int(round(y[i].item()))
                    # Ensure coordinates are within bounds
                    x_int = max(0, min(x_int, self.width - 1))
                    y_int = max(0, min(y_int, self.height - 1))
                    points.append(pointmap[y_int, x_int])

            return torch.stack(points)
        else:
            # No view indices provided - return points from all views
            if x.numel() == 1 and y.numel() == 1:
                x_int, y_int = int(round(x.item())), int(round(y.item()))
                # Ensure coordinates are within bounds
                x_int = max(0, min(x_int, self.width - 1))
                y_int = max(0, min(y_int, self.height - 1))
                return self.pts3d[:, y_int, x_int]

            # For multiple points, we need to handle each one
            points = []
            for i in range(len(x)):
                if i >= self.pts3d.shape[0]:
                    raise ValueError(f"Not enough views in pointmap for coordinate {i}")

                pointmap = self.pts3d[i]

                if use_interpolation:
                    points.append(self.bilinear_interpolate_3d(pointmap, x[i].item(), y[i].item()))
                else:
                    x_int, y_int = int(round(x[i].item())), int(round(y[i].item()))
                    # Ensure coordinates are within bounds
                    x_int = max(0, min(x_int, self.width - 1))
                    y_int = max(0, min(y_int, self.height - 1))
                    points.append(pointmap[y_int, x_int])

            return torch.stack(points)

    def get_confidence_at(
        self,
        x: Union[float, torch.Tensor],
        y: Union[float, torch.Tensor],
        view_idx: Optional[Union[int, torch.Tensor]] = None,
        use_interpolation: bool = False,
    ) -> Union[float, torch.Tensor]:
        """Get the confidence value at the given 2D coordinates.

        Args:
            x: X coordinate(s) (column) - can be a single float or a torch.Tensor
            y: Y coordinate(s) (row) - can be a single float or a torch.Tensor
            view_idx: Index of the view to get the confidence from (if None, returns confidence from all views)
                      Can be a single int, a torch.Tensor, or None
            use_interpolation: Whether to use bilinear interpolation

        Returns:
            Confidence value(s)
        """
        # Convert single values to tensors for consistent handling
        if isinstance(x, (int, float)):
            x = torch.tensor([x])
        if isinstance(y, (int, float)):
            y = torch.tensor([y])

        # Ensure x and y have the same shape
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        # Handle view indices
        if view_idx is not None:
            if isinstance(view_idx, int):
                view_idx = torch.tensor([view_idx])

            # Ensure view indices are within bounds
            if torch.any(view_idx < 0) or torch.any(view_idx >= self.conf.shape[0]):
                raise ValueError(f"View indices must be in range [0, {self.conf.shape[0]-1}]")

            # If x and y are single-element tensors and view_idx is a single-element tensor,
            # we can directly index the confidence map
            if x.numel() == 1 and y.numel() == 1 and view_idx.numel() == 1:
                conf_map = self.conf[view_idx.item()]

                if use_interpolation:
                    # For confidence, we'll use the same bilinear interpolation approach
                    h, w = conf_map.shape

                    # Get the four neighboring points
                    x0, y0 = int(torch.floor(x).item()), int(torch.floor(y).item())
                    x1, y1 = x0 + 1, y0 + 1

                    # Ensure all points are within bounds
                    x0 = max(0, min(x0, w - 1))
                    x1 = max(0, min(x1, w - 1))
                    y0 = max(0, min(y0, h - 1))
                    y1 = max(0, min(y1, h - 1))

                    # Calculate interpolation weights
                    wx = x.item() - x0
                    wy = y.item() - y0

                    # Get confidence values at the four corners
                    c00 = conf_map[y0, x0].item()
                    c01 = conf_map[y0, x1].item()
                    c10 = conf_map[y1, x0].item()
                    c11 = conf_map[y1, x1].item()

                    # Bilinear interpolation formula
                    conf = (
                        (1 - wx) * (1 - wy) * c00
                        + wx * (1 - wy) * c01
                        + (1 - wx) * wy * c10
                        + wx * wy * c11
                    )

                    return conf
                else:
                    x_int, y_int = int(round(x.item())), int(round(y.item()))
                    # Ensure coordinates are within bounds
                    x_int = max(0, min(x_int, self.width - 1))
                    y_int = max(0, min(y_int, self.height - 1))
                    return conf_map[y_int, x_int].item()

            # For multiple points, we need to handle each one
            confidences = []
            for i in range(len(x)):
                idx = view_idx[i].item() if i < len(view_idx) else i
                conf_map = self.conf[idx]

                if use_interpolation:
                    # For confidence, we'll use the same bilinear interpolation approach
                    h, w = conf_map.shape

                    # Get the four neighboring points
                    x0, y0 = int(torch.floor(x[i]).item()), int(torch.floor(y[i]).item())
                    x1, y1 = x0 + 1, y0 + 1

                    # Ensure all points are within bounds
                    x0 = max(0, min(x0, w - 1))
                    x1 = max(0, min(x1, w - 1))
                    y0 = max(0, min(y0, h - 1))
                    y1 = max(0, min(y1, h - 1))

                    # Calculate interpolation weights
                    wx = x[i].item() - x0
                    wy = y[i].item() - y0

                    # Get confidence values at the four corners
                    c00 = conf_map[y0, x0].item()
                    c01 = conf_map[y0, x1].item()
                    c10 = conf_map[y1, x0].item()
                    c11 = conf_map[y1, x1].item()

                    # Bilinear interpolation formula
                    conf = (
                        (1 - wx) * (1 - wy) * c00
                        + wx * (1 - wy) * c01
                        + (1 - wx) * wy * c10
                        + wx * wy * c11
                    )

                    confidences.append(conf)
                else:
                    x_int, y_int = int(round(x[i].item())), int(round(y[i].item()))
                    # Ensure coordinates are within bounds
                    x_int = max(0, min(x_int, self.width - 1))
                    y_int = max(0, min(y_int, self.height - 1))
                    confidences.append(conf_map[y_int, x_int].item())

            return torch.tensor(confidences)
        else:
            # No view indices provided - return confidence from all views
            if x.numel() == 1 and y.numel() == 1:
                x_int, y_int = int(round(x.item())), int(round(y.item()))
                # Ensure coordinates are within bounds
                x_int = max(0, min(x_int, self.width - 1))
                y_int = max(0, min(y_int, self.height - 1))
                return self.conf[:, y_int, x_int]

            # For multiple points, we need to handle each one
            confidences = []
            for i in range(len(x)):
                if i >= self.conf.shape[0]:
                    raise ValueError(f"Not enough views in pointmap for coordinate {i}")

                conf_map = self.conf[i]

                if use_interpolation:
                    # For confidence, we'll use the same bilinear interpolation approach
                    h, w = conf_map.shape

                    # Get the four neighboring points
                    x0, y0 = int(torch.floor(x[i]).item()), int(torch.floor(y[i]).item())
                    x1, y1 = x0 + 1, y0 + 1

                    # Ensure all points are within bounds
                    x0 = max(0, min(x0, w - 1))
                    x1 = max(0, min(x1, w - 1))
                    y0 = max(0, min(y0, h - 1))
                    y1 = max(0, min(y1, h - 1))

                    # Calculate interpolation weights
                    wx = x[i].item() - x0
                    wy = y[i].item() - y0

                    # Get confidence values at the four corners
                    c00 = conf_map[y0, x0].item()
                    c01 = conf_map[y0, x1].item()
                    c10 = conf_map[y1, x0].item()
                    c11 = conf_map[y1, x1].item()

                    # Bilinear interpolation formula
                    conf = (
                        (1 - wx) * (1 - wy) * c00
                        + wx * (1 - wy) * c01
                        + (1 - wx) * wy * c10
                        + wx * wy * c11
                    )

                    confidences.append(conf)
                else:
                    x_int, y_int = int(round(x[i].item())), int(round(y[i].item()))
                    # Ensure coordinates are within bounds
                    x_int = max(0, min(x_int, self.width - 1))
                    y_int = max(0, min(y_int, self.height - 1))
                    confidences.append(conf_map[y_int, x_int].item())

            return torch.tensor(confidences)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pointmap":
        """Create a pointmap from a dictionary.

        Args:
            data: Dictionary representation of the pointmap

        Returns:
            Pointmap instance
        """
        pass

    def bilinear_interpolate_3d(self, pointmap: torch.Tensor, x: float, y: float) -> torch.Tensor:
        """Bilinear interpolation for 3D pointmaps at floating point coordinates.

        Args:
            pointmap: 3D pointmap tensor
            x, y: Floating point coordinates

        Returns:
            Interpolated 3D point
        """
        h, w = pointmap.shape[0:2]

        # Get the four neighboring points
        x0, y0 = int(torch.floor(torch.tensor(x)).item()), int(torch.floor(torch.tensor(y)).item())
        x1, y1 = x0 + 1, y0 + 1

        # Ensure all points are within bounds
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h - 1))

        # Calculate interpolation weights
        wx = x - x0
        wy = y - y0

        # Get 3D points at the four corners
        p00 = pointmap[y0, x0]
        p01 = pointmap[y0, x1]
        p10 = pointmap[y1, x0]
        p11 = pointmap[y1, x1]

        # Bilinear interpolation formula
        point = (
            (1 - wx) * (1 - wy) * p00 + wx * (1 - wy) * p01 + (1 - wx) * wy * p10 + wx * wy * p11
        )

        return point


class TwoViewPointmap(Pointmap):
    """Pointmap for two-view scenarios. (like DUSt3R, MASt3R)

    This class represents a pointmap that contains 3D points from two views in a consistent
    coordinate frame.
    """

    pointmap_type = "two_view"

    def __init__(
        self,
        pts3d: Float[torch.Tensor, "2 H W 3"],
        conf: Optional[Float[torch.Tensor, "2 H W"]] = None,
    ):
        """Initialize the two-view pointmap.

        Args:
            pts3d_1: 3D pointmap for the first view (H×W×3)
            pts3d_2: 3D pointmap for the second view (H×W×3)
            conf_1: Confidence map for the first view (H×W)
            conf_2: Confidence map for the second view (H×W)
        """
        super().__init__(pts3d, conf)

    def export_pts(
        self,
        name: str = "points.ply",
        save_image: bool = False,
        image_name: str = "points.png",
    ) -> None:
        """Export the points to a PLY file.

        Args:
            name: Name of the output PLY file
            save_image: Whether to save an image of the points
            image_name: Name of the output image file
        """
        from utils.dust3r import export_pts as dust3r_export_pts

        # Convert to list of tensors for the export function
        pts = [self.pts3d[0], self.pts3d[1]]
        conf = [self.conf[0], self.conf[1]] if self.conf is not None else None

        dust3r_export_pts(pts, conf=conf, name=name, save_image=save_image, image_name=image_name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwoViewPointmap":
        """Create a two-view pointmap from a dictionary.

        Args:
            data: Dictionary representation of the pointmap

        Returns:
            TwoViewPointmap instance
        """
        pts3d = torch.stack([data["pts3d_1"], data["pts3d_2"]], dim=0)
        if "conf_1" in data and "conf_2" in data:
            conf = torch.stack([data["conf_1"], data["conf_2"]], dim=0)
        else:
            conf = None
        return cls(pts3d, conf)


class TwoViewPointmapWrapper:
    """Wrapper for multiple two-view pointmaps.

    This class holds multiple two-view pointmaps, where each pointmap is accessible via a tuple
    index (i, j) representing the two views.
    """

    pointmap_type = "two_view"

    def __init__(self):
        """Initialize an empty wrapper for two-view pointmaps."""
        self._pointmaps = {}  # Dictionary mapping (i, j) tuples to TwoViewPointmap objects
        self._view_indices = set()  # Set of all view indices used in the pointmaps

    def export_pts(
        self,
        output_dir: str,
        save_images: bool = False,
    ) -> None:
        """Export all pointmaps to PLY files.

        Args:
            output_dir: Directory to save the PLY files
            save_images: Whether to save images of the points
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for (i, j), pointmap in self._pointmaps.items():
            name = os.path.join(output_dir, f"points_{i}_{j}.ply")
            image_name = os.path.join(output_dir, f"points_{i}_{j}.png")
            pointmap.export_pts(name=name, save_image=save_images, image_name=image_name)

    def __getitem__(self, key: Tuple[int, int]) -> TwoViewPointmap:
        """Get a pointmap by its view indices.

        Args:
            key: Tuple of two integers (i, j) representing the view indices

        Returns:
            TwoViewPointmap for the specified view pair

        Raises:
            KeyError: If no pointmap exists for the given view pair
        """
        if key not in self._pointmaps:
            raise KeyError(f"No pointmap exists for view pair {key}")
        return self._pointmaps[key]

    def __setitem__(self, key: Tuple[int, int], value: TwoViewPointmap) -> None:
        """Set a pointmap for a specific view pair.

        Args:
            key: Tuple of two integers (i, j) representing the view indices
            value: TwoViewPointmap to store for this view pair
        """
        if not isinstance(value, TwoViewPointmap):
            raise TypeError("Value must be a TwoViewPointmap")

        self._pointmaps[key] = value
        self._view_indices.add(key[0])
        self._view_indices.add(key[1])

    def __contains__(self, key: Tuple[int, int]) -> bool:
        """Check if a pointmap exists for the given view pair.

        Args:
            key: Tuple of two integers (i, j) representing the view indices

        Returns:
            True if a pointmap exists for the given view pair, False otherwise
        """
        return key in self._pointmaps

    def __len__(self) -> int:
        """Get the number of pointmaps in the wrapper.

        Returns:
            Number of pointmaps
        """
        return len(self._pointmaps)

    def add_pointmap(self, i: int, j: int, pointmap: TwoViewPointmap) -> None:
        """Add a pointmap for a specific view pair.

        Args:
            i: Index of the first view
            j: Index of the second view
            pointmap: TwoViewPointmap to store for this view pair
        """
        self[(i, j)] = pointmap

    def get_pointmap(self, i: int, j: int) -> TwoViewPointmap:
        """Get a pointmap for a specific view pair.

        Args:
            i: Index of the first view
            j: Index of the second view

        Returns:
            TwoViewPointmap for the specified view pair

        Raises:
            KeyError: If no pointmap exists for the given view pair
        """
        return self[(i, j)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the wrapper to a dictionary for serialization.

        Returns:
            Dictionary representation of the wrapper
        """
        return {
            "type": "TwoViewPointmapWrapper",
            "pointmaps": {
                f"{i}_{j}": pointmap.to_dict() for (i, j), pointmap in self._pointmaps.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[Tuple[int, int], Dict[str, Any]]) -> "TwoViewPointmapWrapper":
        """Create a wrapper from a dictionary.

        Args:
            data: Dictionary representation of the wrapper

        Returns:
            TwoViewPointmapWrapper instance
        """
        wrapper = cls()
        for key, pointmap_data in data.items():
            wrapper.add_pointmap(key[0], key[1], TwoViewPointmap.from_dict(pointmap_data))

        return wrapper


class MultiviewPointmap(Pointmap):
    """Pointmap for multiview scenarios.

    This class represents a pointmap that contains 3D points from multiple views in a consistent
    coordinate frame. It can handle any number of views, including the two-view case.
    """

    pointmap_type = "multiview"

    def __init__(
        self,
        pts3d: torch.Tensor,
        conf: Optional[torch.Tensor] = None,
        view_indices: Optional[List[int]] = None,
    ):
        """Initialize the multiview pointmap.

        Args:
            pts3d: 3D pointmap (N×H×W×3) where N is the number of views
            conf: Confidence map (N×H×W)
            view_indices: List of view indices that contributed to this pointmap
            conf_value: Overall confidence value for the pointmap
        """
        # Check if pts3d is a 4D tensor (N×H×W×3)
        if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
            raise ValueError("pts3d must be a 4D tensor of shape N×H×W×3")

        # Get the number of views and resolution
        self.num_views = pts3d.shape[0]
        super().__init__(pts3d, conf)

        self.view_indices = (
            view_indices if view_indices is not None else list(range(self.num_views))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the pointmap to a dictionary for serialization.

        Returns:
            Dictionary representation of the pointmap
        """
        return {
            "type": "MultiviewPointmap",
            "pts3d": self.pts3d,
            "conf": self.conf,
            "view_indices": self.view_indices,
            "conf_value": self.conf_value,
        }

    def export_pts(
        self,
        name: str = "points.ply",
        save_image: bool = False,
        image_name: str = "points.png",
    ) -> None:
        """Export the points to a PLY file.

        Args:
            name: Name of the output PLY file
            save_image: Whether to save an image of the points
            image_name: Name of the output image file
        """
        from utils.dust3r import export_pts as dust3r_export_pts

        # Convert to list of tensors for the export function
        pts = [self.pts3d[i] for i in range(self.num_views)]
        conf = [self.conf[i] for i in range(self.num_views)] if self.conf is not None else None

        dust3r_export_pts(pts, conf=conf, name=name, save_image=save_image, image_name=image_name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiviewPointmap":
        """Create a pointmap from a dictionary.

        Args:
            data: Dictionary representation of the pointmap

        Returns:
            Pointmap instance
        """
        return cls(
            pts3d=data["pts3d"],
            conf=data["conf"],
            view_indices=data["view_indices"],
        )
