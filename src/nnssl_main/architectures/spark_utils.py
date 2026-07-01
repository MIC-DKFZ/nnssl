import sys
import torch
from torch import nn

from einops import rearrange, repeat
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
import torch

from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

allow_ops_in_compiled_graph()

_cur_active: torch.Tensor = None  # B1fff


def _get_active_ex_or_ii(B, D, H, W, device, dtype):
    """
    This probably needs to be adapted. Right now the lowest level defines the mask, but we do it right now at the highest level.
    Otherwise this enforces that the blocks will be quite large in the input (depending on downsampling).
    """
    mask_D, mask_H, mask_W = _cur_active.shape[2:]
    # If we the resolution is smaller than our blocks
    if D < mask_D and H < mask_H and W < mask_W:
        return torch.ones(B, 1, D, H, W, dtype=dtype, device=device)
    # If the resolution is larger than our blocks -?
    d_repeat, h_repeat, w_repeat = D // _cur_active.shape[-3], H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = (
        _cur_active.repeat_interleave(d_repeat, dim=2)
        .repeat_interleave(h_repeat, dim=3)
        .repeat_interleave(w_repeat, dim=4)
        .to(device=device, dtype=dtype)
    )
    return active_ex


def _old_get_active_ex_or_ii(B, D, H, W):
    """
    This probably needs to be adapted. Right now the lowest level defines the mask, but we do it right now at the highest level.
    Otherwise this enforces that the blocks will be quite large in the input (depending on downsampling).
    """
    mask_D, mask_H, mask_W = _cur_active.shape[2:]
    # If we the resolution is smaller than our blocks
    if D < mask_D and H < mask_H and W < mask_W:
        return torch.ones(B, 1, D, H, W, dtype=_cur_active.dtype, device=_cur_active.device)
    # If the resolution is larger than our blocks -?
    d_repeat, h_repeat, w_repeat = D // _cur_active.shape[-3], H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = (
        _cur_active.repeat_interleave(d_repeat, dim=2)
        .repeat_interleave(h_repeat, dim=3)
        .repeat_interleave(w_repeat, dim=4)
    )
    return active_ex


def sp_conv_forward(self, x: torch.Tensor):
    """
    Does the normal conv call, and then masks the output with the active_ex mask.
    """
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(B=x.shape[0], D=x.shape[2], H=x.shape[3], W=x.shape[4], device=x.device, dtype=x.dtype)
    # (BCDHW) *= (B1DHW), mask the output of conv
    return x


# def sp_in_forward(self, x: torch.Tensor):
#     mask = _get_active_ex_or_ii(B=x.shape[0], D=x.shape[2], H=x.shape[3], W=x.shape[4])
#     # active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, di, hi, wi

#     x_1d = rearrange(x, "b c d h w -> b c (d h w)")
#     mask = repeat(mask, "b 1 d h w -> b 1 (d h w)", c=x.shape[1])
#     mask_ids = mask.nonzero(as_tuple=True)

#     ncl = x_1d[mask_ids]
#     ncl = super(type(self), self).forward(ncl)  # use BN1d to normalize this flatten feature `nc`

#     x_postbn = torch.zeros_like(x)
#     x_postbn[mask_ids] = ncl
#     # bcdhw = rearrange(
#     #     x_postbn, "b c (d h w)  -> b c d h w", d=x.shape[2], h=x.shape[3], w=x.shape[4]
#     # )  # reshape the normalized flatten feature back to the original shape
#     return bcdhw


def einops_sp_bn_forward(self, x: torch.Tensor):
    """
    Flatten the input, normalize it, and then reshape it back to the original shape.
    This has to be done to make the masking not affect the norm statistics.
    """
    mask = _get_active_ex_or_ii(
        B=x.shape[0], D=x.shape[2], H=x.shape[3], W=x.shape[4], device=x.device, dtype=x.dtype
    )
    # active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, di, hi, wi
    # ToDo: Test this re-arrange madness.
    #   Should normalize by sample now (not by batch, as we do instance norm and not batchnorm!)
    x_pre_in = rearrange(x, "b c d h w -> b d h w c")
    mask = mask.squeeze(1)
    L = mask.sum(dim=(1, 2, 3))[0]  # Same for all batch elements
    mask_ids = mask.nonzero(as_tuple=True)
    flat_values = x_pre_in[mask_ids]
    ncl = rearrange(flat_values, "(b L) c -> b c L", b=x.shape[0], c=x.shape[1], L=int(L))  # (BCL) -> (BCL)
    ncl = super(type(self), self).forward(ncl)  # use BN1d to normalize this flatten feature `nc`
    ncl = rearrange(ncl, "b c L -> (b L) c")  # (BCL) -> (BCL)

    x_postin = torch.zeros_like(x_pre_in, dtype=x_pre_in.dtype, device=x_pre_in.device)
    x_postin[mask_ids] = ncl
    x_postin = rearrange(x_postin, "b d h w c -> b c d h w")  # (BDHWC) -> (BCDHW)
    # bcdhw = rearrange(
    #     x_postbn, "b c (d h w)  -> b c d h w", d=x.shape[2], h=x.shape[3], w=x.shape[4]
    # )  # reshape the normalized flatten feature back to the original shape
    return x_postin


def old_sp_in_forward(self, x: torch.Tensor):
    """
    Flatten the input, normalize it, and then reshape it back to the original shape.
    This has to be done to make the masking not affect the norm statistics.
    """
    mask = _old_get_active_ex_or_ii(B=x.shape[0], D=x.shape[2], H=x.shape[3], W=x.shape[4])
    # active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, di, hi, wi
    # ToDo: Test this re-arrange madness.
    #   Should normalize by sample now (not by batch, as we do instance norm and not batchnorm!)
    x_pre_in = rearrange(x, "b c d h w -> b d h w c")
    mask = mask.squeeze(1)
    L = mask.sum(dim=(1, 2, 3))[0]  # Same for all batch elements
    mask_ids = mask.nonzero(as_tuple=True)
    flat_values = x_pre_in[mask_ids]
    ncl = rearrange(flat_values, "(b L) c -> b c L", b=x.shape[0], c=x.shape[1], L=int(L))  # (BCL) -> (BCL)
    ncl = super(type(self), self).forward(ncl)  # use BN1d to normalize this flatten feature `nc`
    ncl = rearrange(ncl, "b c L -> (b L) c")  # (BCL) -> (BCL)

    x_postin = torch.zeros_like(x_pre_in, dtype=x_pre_in.dtype, device=x_pre_in.device)
    x_postin[mask_ids] = ncl
    x_postin = rearrange(x_postin, "b d h w c -> b c d h w")  # (BDHWC) -> (BCDHW)
    # bcdhw = rearrange(
    #     x_postbn, "b c (d h w)  -> b c d h w", d=x.shape[2], h=x.shape[3], w=x.shape[4]
    # )  # reshape the normalized flatten feature back to the original shape
    return x_postin


def sp_bn_forward(self, x: torch.Tensor):
    """
    Flatten the input, normalize it, and then reshape it back to the original shape.
    This has to be done to make the masking not affect the norm statistics.
    """
    B, C = x.shape[0], x.shape[1]
    mask = _get_active_ex_or_ii(
        B=x.shape[0], D=x.shape[2], H=x.shape[3], W=x.shape[4], device=x.device, dtype=x.dtype
    )

    x_pre_in = torch.permute(x, (0, 2, 3, 4, 1))
    mask = torch.squeeze(mask, 1)
    L = int(torch.sum(mask, dim=(1, 2, 3))[0])
    mask_ids = torch.nonzero(mask, as_tuple=True)
    flat_values = x_pre_in[mask_ids]

    pre_nlc = torch.reshape(flat_values, (B, L, C))  # (BL)C -> BLC
    pre_ncl = torch.permute(pre_nlc, (0, 2, 1))  # BLC -> BCL
    post_ncl = super(type(self), self).forward(pre_ncl)  # use BN1d to normalize this flatten feature `nc`

    post_nlc = torch.permute(post_ncl, (0, 2, 1))  # BCL -> BLC
    post_ncl = torch.reshape(post_nlc, (B * L, C))  # BLC -> (BL)C

    x_postin = torch.zeros_like(x_pre_in, dtype=x_pre_in.dtype, device=x_pre_in.device)
    x_postin[mask_ids] = post_ncl

    x_postin = torch.permute(x_postin, (0, 4, 1, 2, 3))
    return x_postin


class SparseConv3d(nn.Conv3d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool3d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool3d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class einops_SparseInstanceNorm3d(nn.InstanceNorm1d):
    forward = einops_sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseInstanceNorm3d(nn.InstanceNorm1d):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details

class OldInstanceNorm3d(nn.InstanceNorm1d):
    forward = old_sp_in_forward

# def convert_to_spark_cnn(m: nn.Module, verbose=False, sbn=False):
#     oup = m
#     if isinstance(m, nn.Conv3d):
#         m: nn.Conv3d
#         bias = m.bias is not None
#         oup = SparseConv3d(
#             m.in_channels,
#             m.out_channels,
#             kernel_size=m.kernel_size,
#             stride=m.stride,
#             padding=m.padding,
#             dilation=m.dilation,
#             groups=m.groups,
#             bias=bias,
#             padding_mode=m.padding_mode,
#         )
#         oup.weight.data.copy_(m.weight.data)
#         if bias:
#             oup.bias.data.copy_(m.bias.data)
#     elif isinstance(m, nn.MaxPool3d):
#         m: nn.MaxPool3d
#         oup = SparseMaxPooling(
#             m.kernel_size,
#             stride=m.stride,
#             padding=m.padding,
#             dilation=m.dilation,
#             return_indices=m.return_indices,
#             ceil_mode=m.ceil_mode,
#         )
#     elif isinstance(m, nn.AvgPool3d):
#         m: nn.AvgPool3d
#         oup = SparseAvgPooling(
#             m.kernel_size,
#             m.stride,
#             m.padding,
#             ceil_mode=m.ceil_mode,
#             count_include_pad=m.count_include_pad,
#             divisor_override=m.divisor_override,
#         )
#     elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
#         m: nn.BatchNorm3d
#         oup = (SparseSyncBatchNorm3d if sbn else SparseBatchNorm3d)(
#             m.weight.shape[0],
#             eps=m.eps,
#             momentum=m.momentum,
#             affine=m.affine,
#             track_running_stats=m.track_running_stats,
#         )
#         oup.weight.data.copy_(m.weight.data)
#         oup.bias.data.copy_(m.bias.data)
#         oup.running_mean.data.copy_(m.running_mean.data)
#         oup.running_var.data.copy_(m.running_var.data)
#         oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
#         if hasattr(m, "qconfig"):
#             oup.qconfig = m.qconfig
#     elif isinstance(m, (nn.InstanceNorm3d,)):
#         m: nn.InstanceNorm3d
#         oup = SparseInstanceNorm3d(
#             m.weight.shape[0],
#             eps=m.eps,
#             momentum=m.momentum,
#             affine=m.affine,
#             track_running_stats=m.track_running_stats,
#         )
#         oup.weight.data.copy_(m.weight.data)
#         oup.bias.data.copy_(m.bias.data)
#         if hasattr(m, "qconfig"):
#             oup.qconfig = m.qconfig
#     # elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
#     #     m: nn.LayerNorm
#     #     oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
#     #     oup.weight.data.copy_(m.weight.data)
#     #     oup.bias.data.copy_(m.bias.data)
#     elif isinstance(m, (nn.Conv1d,)):
#         raise NotImplementedError
#     # Right now seems a bit fishy. Seems like infinite recursion.
#     for name, child in m.named_children():
#         oup.add_module(name, convert_to_spark_cnn(child, verbose=verbose, sbn=sbn))
#     del m
#     return oup


def convert_to_spark_cnn(m: nn.Module, verbose=False, sbn=False):
    # Dummy to see if this is what breaks torch.compile or if it's something else.
    oup = m
    if isinstance(m, nn.Conv3d):
        m: nn.Conv3d
        bias = m.bias is not None
        oup = SparseConv3d(
            m.in_channels,
            m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=bias,
            padding_mode=m.padding_mode,
        )
        oup.weight.data.copy_(m.weight.data)
        if bias:
            oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, nn.MaxPool3d):
        m: nn.MaxPool3d
        oup = SparseMaxPooling(
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            return_indices=m.return_indices,
            ceil_mode=m.ceil_mode,
        )
    elif isinstance(m, nn.AvgPool3d):
        m: nn.AvgPool3d
        oup = SparseAvgPooling(
            m.kernel_size,
            m.stride,
            m.padding,
            ceil_mode=m.ceil_mode,
            count_include_pad=m.count_include_pad,
            divisor_override=m.divisor_override,
        )
    elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
        raise NotImplementedError("Only InstanceNorm supported atm.")
    elif isinstance(m, (nn.InstanceNorm3d,)):
        m: nn.InstanceNorm3d
        oup = SparseInstanceNorm3d(
            m.weight.shape[0],
            eps=m.eps,
            momentum=m.momentum,
            affine=m.affine,
            track_running_stats=m.track_running_stats,
        )
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
        if hasattr(m, "qconfig"):
            oup.qconfig = m.qconfig
    elif isinstance(m, (nn.Conv1d,)):
        raise NotImplementedError
    # Right now seems a bit fishy. Seems like infinite recursion.
    for name, child in m.named_children():
        oup.add_module(name, convert_to_spark_cnn(child, verbose=verbose, sbn=sbn))
    del m
    return oup


def convert_to_einops_spark_cnn(m: nn.Module, verbose=False, sbn=False):
    # Dummy to see if this is what breaks torch.compile or if it's something else.
    oup = m
    if isinstance(m, nn.Conv3d):
        m: nn.Conv3d
        bias = m.bias is not None
        oup = SparseConv3d(
            m.in_channels,
            m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=bias,
            padding_mode=m.padding_mode,
        )
        oup.weight.data.copy_(m.weight.data)
        if bias:
            oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, nn.MaxPool3d):
        m: nn.MaxPool3d
        oup = SparseMaxPooling(
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            return_indices=m.return_indices,
            ceil_mode=m.ceil_mode,
        )
    elif isinstance(m, nn.AvgPool3d):
        m: nn.AvgPool3d
        oup = SparseAvgPooling(
            m.kernel_size,
            m.stride,
            m.padding,
            ceil_mode=m.ceil_mode,
            count_include_pad=m.count_include_pad,
            divisor_override=m.divisor_override,
        )
    elif isinstance(m, (nn.InstanceNorm3d,)):
        m: nn.InstanceNorm3d
        oup = einops_SparseInstanceNorm3d(
            m.weight.shape[0],
            eps=m.eps,
            momentum=m.momentum,
            affine=m.affine,
            track_running_stats=m.track_running_stats,
        )
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
        if hasattr(m, "qconfig"):
            oup.qconfig = m.qconfig
    elif isinstance(m, (nn.Conv1d,)):
        raise NotImplementedError
    # Right now seems a bit fishy. Seems like infinite recursion.
    for name, child in m.named_children():
        oup.add_module(name, convert_to_einops_spark_cnn(child, verbose=verbose, sbn=sbn))
    del m
    return oup


if __name__ == "__main__":
    # Test einops_sp_bn_forward vs sp_bn_forward
    # Mask shape should be b, 1, d, h, w
    mask = torch.randint(0, 2, (1, 1, 16, 16, 16))
    mask = repeat(mask, "1 1 d h w -> b 1 d h w", b=4)

    # Test SparseBatchNorm3d
    _cur_active = mask
    sp_in = SparseInstanceNorm3d(3, affine=True)
    einops_sp_in = einops_SparseInstanceNorm3d(3, affine=True)
    old_in = OldInstanceNorm3d(3, affine=True)
    sparse_conv = SparseConv3d(3, 3, 3)

    for j in range(10):
        x = torch.randn(4, 3, 16, 16, 16)
        out_b = einops_sp_in(x)
        out_a = sp_in(x)
        out_o = old_in(x)
        if torch.allclose(out_a, out_b):
            print(f"Passed at {j}")
        if torch.allclose(out_a, out_o):
            print(f"Passed old at {j}")
        else:
            print(f"Failed at {j}")
            diff = torch.sum(torch.abs(out_a - out_b))
            print(f"Diff: {diff}")

    x = torch.randn(4, 3, 16, 16, 16)

    conv = torch.compile(sparse_conv, options={"trace.enabled": True})
    conv(x)
    print("Compiled conv")
    out_a = torch.compile(sp_in)
    out_a(x)
    print("Compiled IN")
    joint = torch.nn.Sequential(*[sparse_conv, sp_in])
    joint_c = torch.compile(joint)
    joint_c(x)
    print("Compiled joint")
    sys.exit(0)
