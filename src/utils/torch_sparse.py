"""Sparse tensor utils take from: https://stackoverflow.com/questions/50666440/column-row-slicing-a-torch-sparse-tensor
and extended myself"""
import torch


def ainb(a, b):
    """Gets mask for elements of a in b, optimized for performance.

    Args:
        a: Tensor of elements to check
        b: Tensor of elements to check against

    Returns:
        Boolean mask where True indicates element in a is also in b
    """
    # Convert inputs to tensors if they aren't already
    a_tensor = a if isinstance(a, torch.Tensor) else torch.tensor(a)
    b_tensor = b if isinstance(b, torch.Tensor) else torch.tensor(b)

    # Use vectorized operations instead of loops
    if len(a_tensor) <= len(b_tensor):
        # Convert b to a set for O(1) lookups
        b_set = set(b_tensor.cpu().numpy().tolist())
        # Use list comprehension for better performance
        mask = torch.tensor(
            [x.item() in b_set for x in a_tensor], dtype=torch.bool, device=a_tensor.device
        )
    else:
        # For each element in b, create a mask where a equals that element, then combine
        mask = torch.zeros(len(a_tensor), dtype=torch.bool, device=a_tensor.device)
        for elem in b_tensor:
            mask = mask | (a_tensor == elem)

    return mask


def slice_torch_sparse_coo_tensor(t, slices):
    """Slice a sparse COOrdinate tensor along specified dimensions.

    Args:
        t: Sparse tensor to slice
        slices: List of indices to select for each dimension

    Returns:
        Sliced sparse tensor
    """
    # Ensure input is coalesced
    t = t.coalesce()
    indices = t.indices()
    values = t.values()

    # Early return if empty
    if indices.shape[1] == 0:
        new_shape = [len(s) for s in slices]
        return torch.sparse_coo_tensor(indices, values, new_shape)

    # Process each dimension
    new_shape = []
    slices_offset = []

    # Create mask for valid indices
    mask = torch.ones(indices.shape[1], dtype=torch.bool, device=indices.device)

    for dim, slice_indices in enumerate(slices):
        # Convert slice to tensor if needed
        slice_tensor = torch.tensor(slice_indices, device=indices.device)
        new_shape.append(len(slice_tensor))

        # Create mask for this dimension and combine with overall mask
        dim_mask = ainb(indices[dim], slice_tensor)
        mask = mask & dim_mask

        # Calculate offsets for remapping indices
        offset_tensor = torch.zeros(t.shape[dim], dtype=torch.long, device=indices.device)
        for i, idx in enumerate(slice_tensor):
            offset_tensor[idx] = i

        # Store offset for later use
        slices_offset.append(offset_tensor)

    # Apply mask to indices and values
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]

    # Remap indices
    new_indices = torch.zeros_like(filtered_indices)
    for dim, offset in enumerate(slices_offset):
        new_indices[dim] = offset[filtered_indices[dim]]

    # Create new sparse tensor
    return torch.sparse_coo_tensor(new_indices, filtered_values, new_shape)


def sparse_diagonal(sparse_tensor):
    """Extract the diagonal of a sparse tensor.

    Args:
        sparse_tensor: Square sparse tensor in COOrdinate format

    Returns:
        Dense tensor containing the diagonal elements
    """
    assert (
        sparse_tensor.layout == torch.sparse_coo
    ), "Input must be a sparse tensor in COOrdinate format"

    # Ensure input is coalesced for efficiency
    sparse_tensor = sparse_tensor.coalesce()

    # Extract the indices and values
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()

    # Determine the size of the sparse tensor (assuming it's square)
    size = sparse_tensor.size(0)

    # Initialize a tensor for the diagonal filled with zeros
    diagonal_values = torch.zeros(size, dtype=values.dtype, device=sparse_tensor.device)

    # Identify the diagonal elements (row index == column index)
    diagonal_mask = indices[0] == indices[1]

    # Get the indices of diagonal elements
    diagonal_indices = indices[0][diagonal_mask]

    # Fill in the diagonal values where they exist
    diagonal_values[diagonal_indices] = values[diagonal_mask]

    return diagonal_values
