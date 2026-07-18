import pytest

from src.utils.dimension import (
    calculate_next_frame_requirement,
    calculate_padded_frame_count,
    compute_scaled_and_target_dims,
)


class TestCalculatePaddedFrameCount:
    def test_zero_frames(self):
        assert calculate_padded_frame_count(0) == 0

    def test_negative_frames(self):
        assert calculate_padded_frame_count(-5) == 0

    def test_single_frame(self):
        assert calculate_padded_frame_count(1) == 1

    def test_just_below_next_boundary(self):
        # n=8 → largest 8k+1 <= 8 is 1
        assert calculate_padded_frame_count(8) == 1

    def test_at_boundary(self):
        assert calculate_padded_frame_count(9) == 9
        assert calculate_padded_frame_count(17) == 17
        assert calculate_padded_frame_count(25) == 25

    def test_between_boundaries(self):
        # 16 is between 9 (8*1+1) and 17 (8*2+1), so result is 9
        assert calculate_padded_frame_count(16) == 9
        assert calculate_padded_frame_count(24) == 17

    def test_result_is_always_8n_plus_1_form(self):
        for n in range(1, 50):
            result = calculate_padded_frame_count(n)
            if result > 0:
                assert (result - 1) % 8 == 0

    def test_result_never_exceeds_input(self):
        for n in range(0, 50):
            assert calculate_padded_frame_count(n) <= n


class TestCalculateNextFrameRequirement:
    def test_below_minimum(self):
        assert calculate_next_frame_requirement(1) == 21
        assert calculate_next_frame_requirement(20) == 21

    def test_at_minimum(self):
        assert calculate_next_frame_requirement(21) == 21

    def test_just_above_boundary(self):
        assert calculate_next_frame_requirement(22) == 29

    def test_at_boundaries(self):
        assert calculate_next_frame_requirement(29) == 29
        assert calculate_next_frame_requirement(37) == 37

    def test_between_boundaries(self):
        # 30 is between 29 and 37
        assert calculate_next_frame_requirement(30) == 37

    def test_result_is_always_8n_plus_5_form(self):
        for n in range(21, 80):
            result = calculate_next_frame_requirement(n)
            assert (result - 5) % 8 == 0

    def test_result_never_less_than_input(self):
        for n in range(1, 80):
            assert calculate_next_frame_requirement(n) >= n


class TestComputeScaledAndTargetDims:
    def test_basic_upscale(self):
        sh, sw, th, tw = compute_scaled_and_target_dims(32, 32, scale=4, multiple=128)
        assert sh == 128 and sw == 128
        assert th == 128 and tw == 128

    def test_target_rounds_up_to_multiple(self):
        # 33 * 4 = 132, next multiple of 128 is 256
        sh, sw, th, tw = compute_scaled_and_target_dims(33, 33, scale=4, multiple=128)
        assert sh == 132 and sw == 132
        assert th == 256 and tw == 256

    def test_target_dims_are_multiples_of_multiple(self):
        for h0, w0 in [(10, 20), (50, 70), (100, 200)]:
            sh, sw, th, tw = compute_scaled_and_target_dims(h0, w0, scale=4, multiple=128)
            assert th % 128 == 0
            assert tw % 128 == 0

    def test_target_dims_never_less_than_scaled(self):
        for h0, w0 in [(10, 20), (50, 70), (128, 256)]:
            sh, sw, th, tw = compute_scaled_and_target_dims(h0, w0, scale=4, multiple=128)
            assert th >= sh
            assert tw >= sw

    def test_asymmetric_input(self):
        sh, sw, th, tw = compute_scaled_and_target_dims(10, 20, scale=2, multiple=16)
        assert sh == 20 and sw == 40
        assert th == 32 and tw == 48

    def test_already_aligned_dims_unchanged(self):
        # 128 * 4 = 512, which is already a multiple of 128
        sh, sw, th, tw = compute_scaled_and_target_dims(128, 128, scale=4, multiple=128)
        assert sh == th == 512
        assert sw == tw == 512

    def test_invalid_zero_height(self):
        with pytest.raises(ValueError):
            compute_scaled_and_target_dims(0, 100)

    def test_invalid_zero_width(self):
        with pytest.raises(ValueError):
            compute_scaled_and_target_dims(100, 0)

    def test_invalid_negative_dims(self):
        with pytest.raises(ValueError):
            compute_scaled_and_target_dims(-10, 100)
