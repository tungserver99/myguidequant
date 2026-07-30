import unittest
from unittest import mock
import importlib.util
from pathlib import Path

import numpy as np
import torch


def _calculate_new_indices(byte_indices, threads_per_warp, offset=0):
    bytes_per_thread = 4
    bytes_per_warp = threads_per_warp * bytes_per_thread
    warp_idx, byte_offsets_within_warp = np.divmod(byte_indices, bytes_per_warp)
    warp_offsets = warp_idx * bytes_per_warp
    thread_indices = byte_indices % threads_per_warp
    byte_offsets_within_thread = byte_offsets_within_warp // threads_per_warp
    byte_offsets_within_thread ^= 3
    return warp_offsets + thread_indices * bytes_per_thread + byte_offsets_within_thread + offset


def _permute_bitmaps(bitmaps):
    _, _, total_bytes = bitmaps.shape
    full_warps_bytes = (total_bytes // 128) * 128
    remaining_bytes_start_idx = full_warps_bytes
    full_warp_byte_indices = np.arange(full_warps_bytes)
    new_full_warp_byte_indices = _calculate_new_indices(full_warp_byte_indices, 32)
    remaining_bytes = total_bytes - full_warps_bytes
    if remaining_bytes:
        remaining_byte_indices = np.arange(remaining_bytes)
        adjusted_threads_per_warp = remaining_byte_indices.size // 4
        new_remaining_byte_indices = _calculate_new_indices(
            remaining_byte_indices, adjusted_threads_per_warp,
            offset=remaining_bytes_start_idx)
        new_byte_indices = np.empty(total_bytes, dtype=np.int64)
        new_byte_indices[:full_warps_bytes] = new_full_warp_byte_indices
        new_byte_indices[full_warps_bytes:] = new_remaining_byte_indices
    else:
        new_byte_indices = new_full_warp_byte_indices
    return bitmaps[:, :, np.argsort(new_byte_indices)]


def _pack_codes(codes, bits):
    flat_codes = codes.reshape(-1)
    bitarray = np.empty((bits, len(flat_codes) // 8), dtype=np.uint8)
    mask = 1 << (bits - 1)
    for bit in range(bits):
        bitarray[bit] = np.packbits((flat_codes & mask).astype(bool))
        mask >>= 1
    bitarray = bitarray.reshape((bits, codes.shape[0], codes.shape[1] // 8))
    packed = np.ascontiguousarray(_permute_bitmaps(bitarray))
    return torch.from_numpy(packed.reshape(-1, 4).view(np.int32).reshape(bits, codes.shape[0], codes.shape[1] // 32))


class AnyPrecisionLinearFallbackTest(unittest.TestCase):
    def test_forward_without_ap_gemv_matches_dense_matmul(self):
        module_path = Path(__file__).resolve().parents[1] / "any_precision" / "modules" / "AnyPrecisionLinear.py"
        spec = importlib.util.spec_from_file_location("anyprecision_linear_module", module_path)
        anyprecision_linear_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(anyprecision_linear_module)

        codes = np.tile(np.arange(8, dtype=np.uint8), 8).reshape(2, 32) % 8
        lut = torch.arange(16, dtype=torch.float16).reshape(2, 8) / 10
        dense_weight = torch.gather(
            lut, 1, torch.from_numpy(codes).long()).to(torch.float32)
        x = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 100
        expected = torch.matmul(x, dense_weight.T)

        with mock.patch.object(anyprecision_linear_module, "ap_gemv", None):
            layer = anyprecision_linear_module.AnyPrecisionLinear(
                in_features=32,
                out_features=2,
                supported_bits=[3],
                bias=False,
                precisions=[3],
                dtype=torch.float16,
            )

        layer.qweight.copy_(_pack_codes(codes, 3))
        layer.lut3.copy_(lut)

        actual = layer(x)

        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
