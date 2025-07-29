import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

def preprocess_points(points, num_bits=11, min_val=None, max_val=None):
    if min_val is None:
        min_val = points.amin(dim=(0,1,2), keepdim=True)
    if max_val is None:
        max_val = points.amax(dim=(0,1,2), keepdim=True)
    
    scaled = (points - min_val) / (max_val - min_val) 
    scaled = scaled * (2**num_bits - 1)
    scaled = scaled.round().clamp(0, 2**num_bits - 1).long()
    return scaled



def sort_point_clouds_hilbert(points, features=None, num_dims=3, num_bits=10, min_val=None, max_val=None):
    B, T, N, C = points.shape
    assert C == num_dims, f"{num_dims}, but {C}"

    if min_val is None:
        min_val = points.amin(dim=(0, 1, 2), keepdim=True)
    if max_val is None:
        max_val = points.amax(dim=(0, 1, 2), keepdim=True)

    scaled = (points - min_val) / (max_val - min_val) 
    scaled = scaled * (2**num_bits - 1)
    scaled_int = scaled.round().clamp(0, 2**num_bits - 1).long() 

    scaled_reshaped = scaled_int.view(B * T * N, C)

    hilbert_ints = encode(scaled_reshaped, num_dims, num_bits) 

    hilbert_ints = hilbert_ints.view(B, T, N)

    sorted_indices = torch.argsort(hilbert_ints, dim=-1) 

    sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, C)

    sorted_points = torch.gather(points, dim=2, index=sorted_indices_expanded) 

    if features is not None:
        F = features.shape[-1]
        sorted_indices_expanded_feat = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, F) 
        sorted_features = torch.gather(features, dim=2, index=sorted_indices_expanded_feat)
    else:
        sorted_features = None

    return sorted_points, sorted_features
def right_shift(binary, k=1, axis=-1):
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted

def gray2binary(gray, axis=-1)
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray

def encode(locs, num_dims, num_bits):
    # Keep around the original shape for later.
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            """
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      """
            % (orig_shape[-1], num_dims)
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a int64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits, num_dims * num_bits)
        )

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    # Now turn these into bits and truncate to num_bits.
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, num_bits):
        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # Convert Gray back to binary.
    hh_bin = gray2binary(gray)

    # Pad back out to 64 bits.
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

    # Convert binary values into uint8s.
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    # Convert uint8s into uint64s.
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()

    return hh_uint64

def binary2gray(binary, axis=-1):
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    gray = torch.logical_xor(binary, shifted)

    return gray


def decode(hilberts, num_dims, num_bits):

    if num_dims * num_bits > 64:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits)
        )

    # Handle the case where we got handed a naked integer.
    hilberts = torch.atleast_1d(hilberts)

    # Keep around the shape for later.
    orig_shape = hilberts.shape
    bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    # Treat each of the hilberts as a s equence of eight uint8.
    # This treats all of the inputs as uint64 and makes things uniform.
    hh_uint8 = (
        hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    )

    # Turn these lists of uints into lists of bits and then truncate to the size
    # we actually need for using Skilling's procedure.
    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -num_dims * num_bits :]
    )

    # Take the sequence of bits and Gray-code it.
    gray = binary2gray(hh_bits)

    # There has got to be a better way to do this.
    # I could index them differently, but the eventual packbits likes it this way.
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    # Iterate backwards through the bits.
    for bit in range(num_bits - 1, -1, -1):
        # Iterate backwards through the dimensions.
        for dim in range(num_dims - 1, -1, -1):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Pad back out to 64 bits.
    extra_dims = 64 - num_bits
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

    # Now chop these up into blocks of 8.
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))

    # Take those blocks and turn them unto uint8s.
    # from IPython import embed; embed()
    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)

    # Finally, treat these as uint64s.
    flat_locs = locs_uint8.view(torch.int64)

    # Return them in the expected shape.
    return flat_locs.reshape((*orig_shape, num_dims))
