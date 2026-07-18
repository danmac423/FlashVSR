import torch

from src.utils.tensor import convert_tensor_to_video, remove_padding, upscale_and_normalize_tensor


class TestConvertTensorToVideo:
    def test_output_shape(self):
        frames = torch.zeros(3, 5, 8, 8)  # (C, N, H, W)
        result = convert_tensor_to_video(frames)
        assert result.shape == (5, 8, 8, 3)  # (N, H, W, C)

    def test_output_dtype_is_float16(self):
        frames = torch.zeros(3, 2, 4, 4)
        result = convert_tensor_to_video(frames)
        assert result.dtype == torch.float16

    def test_minus_one_maps_to_zero(self):
        frames = torch.full((3, 1, 4, 4), -1.0)
        result = convert_tensor_to_video(frames)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_one_maps_to_one(self):
        frames = torch.full((3, 1, 4, 4), 1.0)
        result = convert_tensor_to_video(frames)
        assert torch.allclose(result, torch.ones_like(result))

    def test_zero_maps_to_half(self):
        frames = torch.zeros(3, 1, 4, 4)
        result = convert_tensor_to_video(frames)
        expected = torch.full_like(result, 0.5)
        assert torch.allclose(result, expected)

    def test_output_values_in_unit_range(self):
        frames = torch.rand(3, 4, 8, 8) * 2 - 1  # values in [-1, 1]
        result = convert_tensor_to_video(frames)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestUpscaleAndNormalizeTensor:
    def _make_frame(self, H=8, W=8, C=3, fill=0.5):
        return torch.full((H, W, C), fill)

    def test_output_shape_matches_target(self):
        frame = self._make_frame(8, 8)
        result, _ = upscale_and_normalize_tensor(frame, sh=32, sw=32, th=32, tw=32)
        assert result.shape == (32, 32, 3)

    def test_output_shape_with_padding(self):
        # sh=30, sw=30 → padded to th=32, tw=32
        frame = self._make_frame(8, 8)
        result, padding = upscale_and_normalize_tensor(frame, sh=30, sw=30, th=32, tw=32)
        assert result.shape == (32, 32, 3)

    def test_padding_values_are_non_negative(self):
        frame = self._make_frame(8, 8)
        _, (pad_left, pad_right, pad_top, pad_bottom) = upscale_and_normalize_tensor(
            frame, sh=30, sw=30, th=32, tw=32
        )
        assert pad_left >= 0 and pad_right >= 0
        assert pad_top >= 0 and pad_bottom >= 0

    def test_padding_sums_match_difference(self):
        frame = self._make_frame(8, 8)
        _, (pad_left, pad_right, pad_top, pad_bottom) = upscale_and_normalize_tensor(
            frame, sh=28, sw=24, th=32, tw=32
        )
        assert pad_top + pad_bottom == 32 - 28
        assert pad_left + pad_right == 32 - 24

    def test_no_padding_when_dims_match(self):
        frame = self._make_frame(8, 8)
        _, (pad_left, pad_right, pad_top, pad_bottom) = upscale_and_normalize_tensor(
            frame, sh=32, sw=32, th=32, tw=32
        )
        assert pad_left == pad_right == pad_top == pad_bottom == 0

    def test_output_values_in_normalized_range(self):
        frame = self._make_frame(8, 8, fill=0.5)
        result, _ = upscale_and_normalize_tensor(frame, sh=32, sw=32, th=32, tw=32)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_multichannel_input(self):
        frame = self._make_frame(H=4, W=4, C=4)
        result, _ = upscale_and_normalize_tensor(frame, sh=16, sw=16, th=16, tw=16)
        assert result.shape == (16, 16, 4)


class TestRemovePadding:
    def _make_frames(self, N, H, W, C=3):
        return torch.rand(N, H, W, C)

    def test_removes_height_padding(self):
        frames = self._make_frames(5, 32, 24)
        result = remove_padding(frames, sh=28, sw=24, th=32, tw=24, padding=(0, 0, 2, 2))
        assert result.shape == (5, 28, 24, 3)

    def test_removes_width_padding(self):
        frames = self._make_frames(5, 24, 32)
        result = remove_padding(frames, sh=24, sw=28, th=24, tw=32, padding=(2, 2, 0, 0))
        assert result.shape == (5, 24, 28, 3)

    def test_removes_both_paddings(self):
        frames = self._make_frames(4, 32, 32)
        result = remove_padding(frames, sh=28, sw=28, th=32, tw=32, padding=(2, 2, 2, 2))
        assert result.shape == (4, 28, 28, 3)

    def test_no_padding_returns_original(self):
        frames = self._make_frames(3, 16, 16)
        result = remove_padding(frames, sh=16, sw=16, th=16, tw=16, padding=(0, 0, 0, 0))
        assert result.shape == (3, 16, 16, 3)
        assert torch.equal(result, frames)

    def test_asymmetric_padding(self):
        frames = self._make_frames(2, 32, 32)
        # pad_top=1, pad_bottom=3, pad_left=2, pad_right=2
        result = remove_padding(frames, sh=28, sw=28, th=32, tw=32, padding=(2, 2, 1, 3))
        assert result.shape == (2, 28, 28, 3)

    def test_content_preserved_after_removing_padding(self):
        N, sh, sw, C = 2, 8, 8, 3
        pad_top, pad_bottom, pad_left, pad_right = 2, 2, 2, 2
        th, tw = sh + pad_top + pad_bottom, sw + pad_left + pad_right
        inner = torch.rand(N, sh, sw, C)
        frames = torch.zeros(N, th, tw, C)
        frames[:, pad_top : th - pad_bottom, pad_left : tw - pad_right, :] = inner

        result = remove_padding(
            frames, sh=sh, sw=sw, th=th, tw=tw, padding=(pad_left, pad_right, pad_top, pad_bottom)
        )
        assert torch.allclose(result, inner)
