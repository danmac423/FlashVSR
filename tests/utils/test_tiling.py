import torch

from src.utils.tiling import (
    blend_temporal_overlap,
    calculate_spatial_tile_coords,
    calculate_temporal_tile_ranges,
    create_spatial_blend_mask,
)


class TestCalculateSpatialTileCoords:
    def test_single_tile_fits_exactly(self):
        coords = calculate_spatial_tile_coords(10, 10, tile_size=(10, 10), overlap=0)
        assert coords == [(0, 0, 10, 10)]

    def test_two_rows_no_overlap(self):
        coords = calculate_spatial_tile_coords(20, 10, tile_size=(10, 10), overlap=0)
        assert len(coords) == 2
        assert (0, 0, 10, 10) in coords
        assert (0, 10, 10, 20) in coords

    def test_two_cols_no_overlap(self):
        coords = calculate_spatial_tile_coords(10, 20, tile_size=(10, 10), overlap=0)
        assert len(coords) == 2
        assert (0, 0, 10, 10) in coords
        assert (10, 0, 20, 10) in coords

    def test_with_overlap_produces_more_tiles(self):
        coords_no_overlap = calculate_spatial_tile_coords(20, 10, tile_size=(10, 10), overlap=0)
        coords_with_overlap = calculate_spatial_tile_coords(20, 10, tile_size=(10, 10), overlap=5)
        assert len(coords_with_overlap) > len(coords_no_overlap)

    def test_all_coords_within_bounds(self):
        height, width = 50, 70
        coords = calculate_spatial_tile_coords(height, width, tile_size=(20, 20), overlap=5)
        for x1, y1, x2, y2 in coords:
            assert x1 >= 0 and y1 >= 0
            assert x2 <= width and y2 <= height

    def test_tile_dims_equal_tile_size_or_clamped(self):
        coords = calculate_spatial_tile_coords(50, 70, tile_size=(20, 20), overlap=5)
        for x1, y1, x2, y2 in coords:
            assert (x2 - x1) <= 20
            assert (y2 - y1) <= 20

    def test_coverage_includes_last_row_col(self):
        # The last tile should reach the bottom-right corner
        height, width = 25, 35
        coords = calculate_spatial_tile_coords(height, width, tile_size=(20, 20), overlap=5)
        max_x2 = max(x2 for x1, y1, x2, y2 in coords)
        max_y2 = max(y2 for x1, y1, x2, y2 in coords)
        assert max_x2 == width
        assert max_y2 == height

    def test_four_quadrant_grid(self):
        coords = calculate_spatial_tile_coords(20, 20, tile_size=(10, 10), overlap=0)
        assert len(coords) == 4


class TestCalculateTemporalTileRanges:
    def test_single_chunk_exact(self):
        chunks = calculate_temporal_tile_ranges(10, chunk_size=10, overlap=0)
        assert chunks == [(0, 10)]

    def test_no_overlap_multiple_chunks(self):
        chunks = calculate_temporal_tile_ranges(30, chunk_size=10, overlap=0)
        assert chunks == [(0, 10), (10, 20), (20, 30)]

    def test_with_overlap(self):
        chunks = calculate_temporal_tile_ranges(20, chunk_size=10, overlap=5)
        assert chunks == [(0, 10), (5, 15), (10, 20)]

    def test_last_chunk_capped_at_total_frames(self):
        chunks = calculate_temporal_tile_ranges(25, chunk_size=10, overlap=0)
        last_start, last_end = chunks[-1]
        assert last_end == 25

    def test_chunks_start_at_zero(self):
        chunks = calculate_temporal_tile_ranges(20, chunk_size=10, overlap=2)
        assert chunks[0][0] == 0

    def test_overlap_produces_more_chunks_than_no_overlap(self):
        chunks_no_overlap = calculate_temporal_tile_ranges(20, chunk_size=10, overlap=0)
        chunks_with_overlap = calculate_temporal_tile_ranges(20, chunk_size=10, overlap=5)
        assert len(chunks_with_overlap) > len(chunks_no_overlap)

    def test_all_end_frames_within_bounds(self):
        chunks = calculate_temporal_tile_ranges(17, chunk_size=8, overlap=3)
        for start, end in chunks:
            assert end <= 17

    def test_consecutive_chunks_overlap_by_correct_amount(self):
        overlap = 4
        chunks = calculate_temporal_tile_ranges(20, chunk_size=10, overlap=overlap)
        for i in range(len(chunks) - 1):
            _, end_prev = chunks[i]
            start_next, _ = chunks[i + 1]
            assert end_prev - start_next == overlap


class TestCreateSpatialBlendMask:
    def test_output_shape(self):
        mask = create_spatial_blend_mask((10, 20), overlap=2)
        assert mask.shape == (1, 10, 20, 1)

    def test_values_in_unit_range(self):
        mask = create_spatial_blend_mask((16, 16), overlap=4)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_center_is_one(self):
        H, W, overlap = 20, 20, 4
        mask = create_spatial_blend_mask((H, W), overlap=overlap)
        center = mask[0, overlap:-overlap, overlap:-overlap, 0]
        assert torch.all(center == 1.0)

    def test_edges_less_than_center(self):
        mask = create_spatial_blend_mask((20, 20), overlap=4)
        center_val = mask[0, 10, 10, 0].item()
        edge_val = mask[0, 0, 0, 0].item()
        assert edge_val < center_val

    def test_mask_dtype_is_float(self):
        mask = create_spatial_blend_mask((10, 10), overlap=2)
        assert mask.dtype == torch.float32

    def test_asymmetric_size(self):
        mask = create_spatial_blend_mask((8, 16), overlap=2)
        assert mask.shape == (1, 8, 16, 1)


class TestBlendTemporalOverlap:
    def _make_frames(self, n_frames, fill_value=1.0):
        return torch.full((n_frames, 4, 4, 3), fill_value)

    def test_zero_overlap_returns_current(self):
        prev = self._make_frames(5, fill_value=0.0)
        curr = self._make_frames(5, fill_value=1.0)
        result = blend_temporal_overlap(prev, curr, overlap=0)
        assert torch.equal(result, curr)

    def test_none_prev_returns_current(self):
        curr = self._make_frames(5, fill_value=1.0)
        result = blend_temporal_overlap(None, curr, overlap=3)
        assert torch.equal(result, curr)

    def test_blended_shape(self):
        prev = self._make_frames(5, fill_value=0.0)
        curr = self._make_frames(5, fill_value=1.0)
        result = blend_temporal_overlap(prev, curr, overlap=3)
        assert result.shape == (3, 4, 4, 3)

    def test_first_blended_frame_close_to_prev(self):
        prev = self._make_frames(5, fill_value=0.0)
        curr = self._make_frames(5, fill_value=1.0)
        result = blend_temporal_overlap(prev, curr, overlap=4)
        # blend_weight[0]=1: result[0] ≈ prev[-4]*1 + curr[0]*0 = 0.0
        assert torch.allclose(result[0], torch.zeros_like(result[0]))

    def test_last_blended_frame_close_to_current(self):
        prev = self._make_frames(5, fill_value=0.0)
        curr = self._make_frames(5, fill_value=1.0)
        result = blend_temporal_overlap(prev, curr, overlap=4)
        # blend_weight[-1] approaches 0: result[-1] is mostly curr
        # linspace(1, 0, 4) = [1, 0.667, 0.333, 0.0]
        # result[-1] = 0.0 * 0.0 + 1.0 * (1-0.0) = 1.0
        assert torch.allclose(result[-1], torch.ones_like(result[-1]))

    def test_overlap_larger_than_chunks_clamps(self):
        prev = self._make_frames(3)
        curr = self._make_frames(3)
        # overlap=10 > both chunk sizes (3), actual_overlap should clamp to 3
        result = blend_temporal_overlap(prev, curr, overlap=10)
        assert result.shape[0] == 3

    def test_negative_overlap_returns_current(self):
        prev = self._make_frames(5)
        curr = self._make_frames(5)
        result = blend_temporal_overlap(prev, curr, overlap=-1)
        assert torch.equal(result, curr)
