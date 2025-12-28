import torch
from torch.nn import functional as F
from torch.autograd import Function
from einops import rearrange, repeat
from grouped_gemm.backend import gmm  # type: ignore
import cutlass  # type: ignore
from grouped_gemm.ops import permute, unpermute  # type: ignore
from typing import List, Tuple, Optional

plan = cutlass.op.GroupedGemm(
    element=torch.bfloat16, layout=cutlass.LayoutType.RowMajor
)
cutlass_gmm = cutlass.emit.pytorch(
    plan.construct(), name="cutlass_gmm", cc=plan.cc, sourcedir="out", jit=True
)


class GroupedGEMM(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        block: torch.Tensor,
        bias: Optional[torch.Tensor],
        gate: torch.Tensor,
        index: torch.Tensor,
        config,
        type: str,
    ):
        """
        attn:
            x: [s, d], block: [l * h, d, e], bias: [l * h, (1), e],
            gate: [s * h * k, 1], index: [s * h * k]
        attn_o:
            x: [h, s, e], block: [l * h, e, d], bias: [l * h, (1), d],
            gate: [s * h * k, 1], index: [s * h * k]
        ffn:
            x: [s, d], block: [l * h, 3, d, e], bias: [l * h, (1), d],
            gate: [s * k, 1], index: [s * k]
        """

        dtype = x.dtype
        device = x.device
        option = config.grouped_gemm_option
        gmm_option = option == "gmm"
        c_gmm_option = option == "cutlass_gmm"
        pad_bmm_option = option == "pad_bmm"
        block_num = block.shape[0]

        if type == "attn":
            head_num = config.layer_attn_expert_num
            top_k_num = config.attn_top_k_num
            x_repeat = repeat(x, "s d -> (s h k) d", h=head_num, k=top_k_num)  # TODO
        if type == "attn_o":
            head_num = config.layer_attn_o_expert_num
            top_k_num = config.attn_o_top_k_num
            x_repeat = repeat(x, "h s e -> (s h k) e", k=top_k_num)
        if type == "ffn":
            head_num = 1
            top_k_num = config.ffn_top_k_num
            x_repeat = repeat(x, "s d -> (s k) d", k=top_k_num)

        if pad_bmm_option:
            new_index, freq_idx_s, num_list, freq_list, split_list, freq_len = (
                pad_bmm_preprocess(index, block_num)
            )
            num = num_list[-1]
            index_s = torch.argsort(new_index, stable=False)
            rank_w_pad = torch.empty_like(index_s).scatter_(
                0, index_s, torch.arange(new_index.shape[0], device=device)
            )
            rank = rank_w_pad[: index.shape[0]]
            info = (type, index, index_s, freq_idx_s, rank, freq_len, block.shape)
            list_info = (num_list, freq_list, split_list)
            result = pad_bmm_prepare(x_repeat, info, list_info, "forward")
            len_dim_list, index_s_list, freq_idx_s_list, x_list = result
            index_len, _, y_dim = len_dim_list
            index_s1, index_s2, index_s3 = index_s_list
            freq_idx_s1, freq_idx_s2, freq_idx_s3 = freq_idx_s_list
            x1, x2, x3 = x_list
            del x_repeat, result, x_list
            y1, y11, y12 = pad_bmm_forward(x1, block, bias, type, freq_idx_s1)
            del x1
            y2, y21, y22 = pad_bmm_forward(x2, block, bias, type, freq_idx_s2)
            del x2
            y3, y31, y32 = pad_bmm_forward(x3, block, bias, type, freq_idx_s3)
            del x3
            y_s1 = (y11, y21, y31)
            y_s2 = (y12, y22, y32)
            y = torch.empty(num, y_dim, dtype=dtype, device=device)
            y[index_s1] = y1
            y[index_s2] = y2
            y[index_s3] = y3
            del y1, y2, y3
            y = y[:index_len]
        else:
            index_s = torch.argsort(index, stable=False)
            x_s = x_repeat[index_s, :]
            del x_repeat
            if gmm_option and x_s.dtype != torch.bfloat16:
                x_s = x_s.to(torch.bfloat16)
            index_freq = torch.bincount(index, minlength=block.shape[0])
            if gmm_option:
                index_freq = index_freq.to("cpu")
            if c_gmm_option:
                idx_freq = index_freq.tolist()
            if type == "ffn":
                if gmm_option:
                    y_s1 = gmm(x_s, block[:, 0], index_freq)
                    y_s2 = gmm(x_s, block[:, 1], index_freq)
                    del x_s
                    y_s3 = F.silu(y_s1) * y_s2
                    y_s = gmm(y_s3, block[:, 2], index_freq, trans_b=True)
                    del y_s3
                if c_gmm_option:
                    x_s = torch.split(x_s, idx_freq, 0)
                    block_1 = torch.split(block[:, 0], 1, 0)
                    block_2 = torch.split(block[:, 1], 1, 0)
                    block_3 = torch.split(block[:, 2].transpose(1, 2), 1, 0)
                    y_s1 = cutlass_gmm.run(x_s, block_1)
                    y_s2 = cutlass_gmm.run(x_s, block_2)
                    del x_s, block_1, block_2
                    y_s1 = torch.cat(y_s1, 0)
                    y_s2 = torch.cat(y_s2, 0)
                    y_s3 = F.silu(y_s1) * y_s2
                    y_s3 = torch.split(y_s3, idx_freq, 0)
                    y_s = cutlass_gmm.run(y_s3, block_3)
                    y_s = torch.cat(y_s, 0)
                    del y_s3, block_3
            else:
                if gmm_option:
                    y_s = gmm(x_s, block, index_freq)
                    del x_s
                if c_gmm_option:
                    x_s = torch.split(x_s, idx_freq, 0)
                    block_split = torch.split(block, 1, 0)
                    y_s = cutlass_gmm.run(x_s, block_split)
                    del x_s, block_split
                    y_s = torch.cat(y_s, 0)
            y = torch.empty_like(y_s)
            y[index_s, :] = y_s
            del y_s
            if config.expert_bias:
                y += bias[index]
        y_output = y * gate
        if type != "attn":
            del y
        # y_output: attn: [s * h * k, e], attn_o: [s * h * k, d], ffn: [s * k, d]
        if type == "attn":
            y_output = rearrange(
                y_output, "(s h k) e -> s k (h e)", h=head_num, k=top_k_num
            )
            if top_k_num == 1:
                y_output = y_output.squeeze(1)
                if y_output.dtype != dtype:
                    y_output = y_output.to(dtype)
            else:
                y_output = torch.sum(y_output, 1, dtype=dtype)
        else:
            y_output = rearrange(
                y_output, "(s hk) d -> s hk d", hk=head_num * top_k_num
            )
            y_output = torch.sum(y_output, 1, dtype=dtype)

        if pad_bmm_option:
            if type == "attn":
                ctx.save_for_backward(
                    x, block, bias, gate, index_s, freq_idx_s, rank, y
                )
            if type == "attn_o":
                ctx.save_for_backward(x, block, bias, gate, index_s, freq_idx_s, rank)
            if type == "ffn":
                ctx.save_for_backward(
                    x, block, bias, gate, index_s, freq_idx_s, rank, *y_s1, *y_s2
                )
            ctx.info = (config, type, head_num, top_k_num, freq_len)
            ctx.list_info = (num_list, freq_list, split_list)
        else:
            if type == "attn":
                ctx.save_for_backward(
                    x, block, bias, gate, index, index_s, index_freq, y
                )
            if type == "attn_o":
                ctx.save_for_backward(x, block, bias, gate, index, index_s, index_freq)
            if type == "ffn":
                ctx.save_for_backward(
                    x, block, bias, gate, index, index_s, index_freq, y_s1, y_s2
                )
            ctx.info = (config, type, head_num, top_k_num)
        return y_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        option = ctx.info[0].grouped_gemm_option
        gmm_option = option == "gmm"
        c_gmm_option = option == "cutlass_gmm"
        pad_bmm_option = option == "pad_bmm"
        if pad_bmm_option:
            config, type, head_num, top_k_num, freq_len = ctx.info
            num_list, freq_list, split_list = ctx.list_info
            num_1, num_2, num = num_list
            freq_1, freq_2, freq_3 = freq_list
        else:
            config, type, head_num, top_k_num = ctx.info
        dtype = grad_output.dtype
        device = grad_output.device
        router_dtype = config.router_dtype
        grad_block_option = not config.expert_grad_detach
        grad_block = None
        grad_bias = None

        if type == "attn":
            if pad_bmm_option:
                x, block, bias, gate, index_s, freq_idx_s, rank, y = ctx.saved_tensors
            else:
                x, block, bias, gate, index, index_s, index_freq, y = ctx.saved_tensors
            x_repeat = repeat(x, "s d -> (s h k) d", h=head_num, k=top_k_num)
        if type == "attn_o":
            if pad_bmm_option:
                x, block, bias, gate, index_s, freq_idx_s, rank = ctx.saved_tensors
            else:
                x, block, bias, gate, index, index_s, index_freq = ctx.saved_tensors
            x_repeat = repeat(x, "h s e -> (s h k) e", k=top_k_num)
        if type == "ffn":
            if pad_bmm_option:
                x, block, bias, gate, index_s, freq_idx_s, rank = ctx.saved_tensors[:7]
                y11, y21, y31, y12, y22, y32 = ctx.saved_tensors[7:]
            else:
                x, block, bias, gate, index, index_s, index_freq, y_s1, y_s2 = (
                    ctx.saved_tensors
                )
            x_repeat = repeat(x, "s d -> (s k) d", k=top_k_num)

        if type == "attn":
            grad_o = repeat(
                grad_output, "s (h e) -> (s h k) e", h=head_num, k=top_k_num
            )
        else:
            grad_o = repeat(grad_output, "s d -> (s h k) d", h=head_num, k=top_k_num)

        if pad_bmm_option:
            info = (type, index, index_s, freq_idx_s, rank, freq_len)
            list_info = (num_list, freq_list, split_list)
            result = pad_bmm_prepare(x_repeat, info, list_info, "backward")
            len_dim_list, index_s_list, freq_idx_s_list, x_list = result
            index_len, x_dim, y_dim = len_dim_list
            index_s1, index_s2, index_s3 = index_s_list
            freq_idx_s1, freq_idx_s2, freq_idx_s3 = freq_idx_s_list
            x1, x2, x3 = x_list
            del result, x_repeat, x_list

            if type == "attn":
                grad_gate = grad_o * y
                del y
                grad_gate = torch.sum(grad_gate, -1, dtype=router_dtype, keepdim=True)
            if type == "attn_o":
                y1, _, _ = pad_bmm_forward(x1, block, bias, type, freq_idx_s1)
                y2, _, _ = pad_bmm_forward(x2, block, bias, type, freq_idx_s2)
                y3, _, _ = pad_bmm_forward(x3, block, bias, type, freq_idx_s3)
            if type == "ffn":
                y11_silu = F.silu(y11)
                y21_silu = F.silu(y21)
                y31_silu = F.silu(y31)
                y13 = y11_silu * y12
                y23 = y21_silu * y22
                y33 = y31_silu * y32
                y1 = torch.bmm(y13, block[freq_idx_s1][:, 2].transpose(1, 2))
                y2 = torch.bmm(y23, block[freq_idx_s2][:, 2].transpose(1, 2))
                y3 = torch.bmm(y33, block[freq_idx_s3][:, 2].transpose(1, 2))
                if config.expert_bias:
                    y1 += bias[freq_idx_s1]
                    y2 += bias[freq_idx_s2]
                    y3 += bias[freq_idx_s3]

            grad_o_s = torch.zeros(num, y_dim, device=device, dtype=dtype)
            grad_o_s[rank] = grad_o
            del grad_o
            grad_o1 = grad_o_s[:num_1].reshape(-1, freq_1, y_dim)
            grad_o2 = grad_o_s[num_1 : num_1 + num_2].reshape(-1, freq_2, y_dim)
            grad_o3 = grad_o_s[num_1 + num_2 :].reshape(-1, freq_3, y_dim)

            if type != "attn":
                grad_gate_1 = torch.sum(grad_o1 * y1, -1, dtype=router_dtype)
                del y1
                grad_gate_2 = torch.sum(grad_o2 * y2, -1, dtype=router_dtype)
                del y2
                grad_gate_3 = torch.sum(grad_o3 * y3, -1, dtype=router_dtype)
                del y3
                grad_gate = torch.empty(num, device=device, dtype=router_dtype)
                grad_gate[index_s1] = grad_gate_1.flatten()
                grad_gate[index_s2] = grad_gate_2.flatten()
                grad_gate[index_s3] = grad_gate_3.flatten()
                grad_gate = grad_gate[:index_len].unsqueeze(-1)

            gate_zone = torch.zeros(num, device=device, dtype=dtype)
            gate_zone[rank] = gate.flatten()
            gate_1 = gate_zone[:num_1].reshape(-1, freq_1, 1)
            gate_2 = gate_zone[num_1 : num_1 + num_2].reshape(-1, freq_2, 1)
            gate_3 = gate_zone[num_1 + num_2 :].reshape(-1, freq_3, 1)
            grad_y1 = grad_o1 * gate_1
            del grad_o1
            grad_y2 = grad_o2 * gate_2
            del grad_o2
            grad_y3 = grad_o3 * gate_3
            del grad_o3

            if config.expert_bias:
                grad_bias_1 = torch.sum(grad_y1, 1, keepdim=True, dtype=dtype)
                grad_bias_2 = torch.sum(grad_y2, 1, keepdim=True, dtype=dtype)
                grad_bias_3 = torch.sum(grad_y3, 1, keepdim=True, dtype=dtype)
                grad_bias = torch.zeros_like(bias)
                grad_bias[freq_idx_s1] = grad_bias_1
                grad_bias[freq_idx_s2] = grad_bias_2
                grad_bias[freq_idx_s3] = grad_bias_3

            if type == "ffn":
                xy_list_1 = [x1, y11, y11_silu, y12, y13, grad_y1]
                xy_list_2 = [x2, y21, y21_silu, y22, y23, grad_y2]
                xy_list_3 = [x3, y31, y31_silu, y32, y33, grad_y3]
            else:
                xy_list_1 = [x1, grad_y1]
                xy_list_2 = [x2, grad_y2]
                xy_list_3 = [x3, grad_y3]

            grad_x1, grad_block_1 = pad_bmm_backward(
                xy_list_1, block[freq_idx_s1], type, grad_block_option
            )
            grad_x2, grad_block_2 = pad_bmm_backward(
                xy_list_2, block[freq_idx_s2], type, grad_block_option
            )
            grad_x3, grad_block_3 = pad_bmm_backward(
                xy_list_3, block[freq_idx_s3], type, grad_block_option
            )

            grad_x = torch.empty(num, x_dim, dtype=dtype, device=device)
            grad_x[index_s1] = grad_x1.flatten(0, 1)
            grad_x[index_s2] = grad_x2.flatten(0, 1)
            grad_x[index_s3] = grad_x3.flatten(0, 1)
            grad_x = grad_x[:index_len]
            del grad_x1, grad_x2, grad_x3

            if grad_block_option:
                grad_block = torch.zeros_like(block)
                if type == "ffn":
                    for i in range(3):
                        grad_block[freq_idx_s1, i] = grad_block_1[i]
                        grad_block[freq_idx_s2, i] = grad_block_2[i]
                        grad_block[freq_idx_s3, i] = grad_block_3[i]
                else:
                    grad_block[freq_idx_s1] = grad_block_1
                    grad_block[freq_idx_s2] = grad_block_2
                    grad_block[freq_idx_s3] = grad_block_3
                del grad_block_1, grad_block_2, grad_block_3
        else:
            x_s = x_repeat[index_s, :]
            del x_repeat
            if gmm_option and x_s.dtype != torch.bfloat16:
                x_s = x_s.to(torch.bfloat16)
            if type == "attn_o":
                if gmm_option:
                    y_s = gmm(x_s, block, index_freq)
                if c_gmm_option:
                    block_split = torch.split(block, 1, 0)
                    x_st = torch.split(x_s.t(), index_freq, 1)
                    x_s = torch.split(x_s, index_freq, 0)
                    y_s = cutlass_gmm.run(x_s, block_split)
                    y_s = torch.cat(y_s, 0)
            if type == "ffn":
                y_s1_silu = F.silu(y_s1)
                y_s3 = y_s1_silu * y_s2
                if gmm_option:
                    y_s = gmm(y_s3, block[:, 2], index_freq, trans_b=True)
                if c_gmm_option:
                    y_s3 = torch.split(y_s3, index_freq, 0)
                    block_3_t = torch.split(block[:, 2].transpose(1, 2), 1, 0)
                    y_s = cutlass_gmm.run(y_s3, block_3_t)
                    y_s = torch.cat(y_s, 0)
            if type != "attn":
                if config.expert_bias:
                    y_s += bias[index_s]

            grad_o_s = grad_o[index_s, :]
            del grad_o
            grad_gate_s = torch.sum(
                grad_o_s * y_s, -1, dtype=router_dtype, keepdim=True
            )
            grad_gate = torch.empty_like(grad_gate_s)
            grad_gate[index_s, :] = grad_gate_s
            gate_s = torch.empty_like(gate)
            gate_s[index_s, :] = gate
            grad_y_s = grad_o_s * gate_s
            del grad_o_s
            if config.expert_bias:
                grad_bias = torch.zeros(block.shape[0], grad_y_s.shape[-1], dtype=dtype)
                s_index_r = repeat(index[index_s], "n -> n d", d=grad_y_s.shape[-1])
                grad_bias.scatter_add_(0, s_index_r, grad_y_s)

            if gmm_option and grad_y_s.dtype != torch.bfloat16:
                grad_y_s = grad_y_s.to(torch.bfloat16)

            if type == "ffn":
                if gmm_option:
                    grad_y_s3 = gmm(grad_y_s, block[:, 2], index_freq)
                    if grad_block_option:
                        grad_block_3 = gmm(grad_y_s, y_s3, index_freq, trans_a=True)
                if c_gmm_option:
                    if grad_block_option:
                        grad_y_st = torch.split(grad_y_s.t(), index_freq, 1)
                        grad_block_3 = cutlass_gmm.run(grad_y_st, y_s3)
                        grad_block_3 = torch.stack(grad_block_3, 0)
                        del grad_y_st
                    grad_y_s = torch.split(grad_y_s, index_freq, 0)
                    block_3 = torch.split(block[:, 2], 1, 0)
                    grad_y_s3 = cutlass_gmm.run(grad_y_s, block_3)
                    grad_y_s3 = torch.cat(grad_y_s3, 0)
                del grad_y_s, y_s3

                grad_y_s1_silu = grad_y_s3 * y_s2
                grad_y_s2 = grad_y_s3 * y_s1_silu
                del grad_y_s3, y_s2, y_s1_silu
                y_s1_sig = torch.sigmoid(y_s1)
                grad_y_s1 = grad_y_s1_silu * (
                    y_s1_sig + y_s1 * y_s1_sig * (1 - y_s1_sig)
                )
                del grad_y_s1_silu, y_s1, y_s1_sig

                if gmm_option:
                    if grad_block_option:
                        grad_block_1 = gmm(x_s, grad_y_s1, index_freq, trans_a=True)
                        grad_block_2 = gmm(x_s, grad_y_s2, index_freq, trans_a=True)
                    del x_s
                    grad_x_s1 = gmm(grad_y_s1, block[:, 0], index_freq, trans_b=True)
                    grad_x_s2 = gmm(grad_y_s2, block[:, 1], index_freq, trans_b=True)
                if c_gmm_option:
                    grad_y_s1 = torch.split(grad_y_s1, index_freq, 0)
                    grad_y_s2 = torch.split(grad_y_s2, index_freq, 0)
                    if grad_block_option:
                        grad_block_1 = cutlass_gmm.run(x_st, grad_y_s1)
                        grad_block_2 = cutlass_gmm.run(x_st, grad_y_s2)
                        grad_block_1 = torch.stack(grad_block_1, 0)
                        grad_block_2 = torch.stack(grad_block_2, 0)
                    del x_s, x_st
                    block_1 = torch.split(block[:, 0].transpose(1, 2), 1, 0)
                    block_2 = torch.split(block[:, 1].transpose(1, 2), 1, 0)
                    grad_x_s1 = cutlass_gmm.run(grad_y_s1, block_1)
                    grad_x_s2 = cutlass_gmm.run(grad_y_s2, block_2)
                    grad_x_s1 = torch.cat(grad_x_s1, 0)
                    grad_x_s2 = torch.cat(grad_x_s2, 0)
                del grad_y_s1, grad_y_s2

                grad_x_s = grad_x_s1 + grad_x_s2
                del grad_x_s1, grad_x_s2
                if grad_block_option:
                    grad_block = torch.stack(
                        (grad_block_1, grad_block_2, grad_block_3), 1
                    )
                    del grad_block_1, grad_block_2, grad_block_3
            else:
                if gmm_option:
                    if grad_block_option:
                        grad_block = gmm(x_s, grad_y_s, index_freq, trans_a=True)
                    grad_x_s = gmm(grad_y_s, block, index_freq, trans_b=True)
                if c_gmm_option:
                    grad_y_s = torch.split(grad_y_s, index_freq, 0)
                    if grad_block_option:
                        grad_block = cutlass_gmm.run(x_st, grad_y_s)
                        grad_block = torch.stack(grad_block, 0)
                    block_t = torch.split(block.transpose(1, 2), 1, 0)
                    grad_x_s = cutlass_gmm.run(grad_y_s, block_t)
                    grad_x_s = torch.cat(grad_x_s, 0)
                    del x_st
                del x_s, grad_y_s

            grad_x = torch.empty_like(grad_x_s)
            grad_x[index_s, :] = grad_x_s
            del grad_x_s

        if type == "attn":
            grad_x = rearrange(grad_x, "(s hk) d -> s hk d", hk=head_num * top_k_num)
            grad_x = torch.sum(grad_x, 1, dtype=dtype)
        if type == "attn_o":
            grad_x = rearrange(grad_x, "(s h k) e -> h s k e", h=head_num, k=top_k_num)
            if top_k_num == 1:
                grad_x = grad_x.squeeze(2)
                if grad_x.dtype != dtype:
                    grad_x = grad_x.to(dtype)
            else:
                grad_x = torch.sum(grad_x, 2, dtype=dtype)
        if type == "ffn":
            grad_x = rearrange(grad_x, "(s k) d -> s k d", k=top_k_num)
            grad_x = torch.sum(grad_x, 1, dtype=dtype)

        return grad_x, grad_block, grad_bias, grad_gate, None, None, None


def pad_bmm_preprocess(index: torch.Tensor, block_num: int):
    device = index.device
    index_freq = torch.bincount(index, minlength=block_num)
    freq_len = torch.sum(index_freq > 0).item()
    freq_s, freq_idx_s = torch.sort(index_freq, descending=True, stable=False)
    seq_1 = torch.arange(32, device=device)
    seq_2 = torch.arange(freq_len, device=device)
    grid_1, grid_2 = torch.meshgrid(seq_1, seq_2, indexing="ij")
    mask = (grid_1 > 0) & (grid_2 > grid_1) & (grid_2 < freq_len)

    freq_grid_1 = freq_s[0]
    freq_grid_2 = freq_s[grid_1]
    freq_grid_3 = freq_s[grid_2]
    area_1 = freq_grid_1 * grid_1
    area_2 = freq_grid_2 * (grid_2 - grid_1)
    area_3 = freq_grid_3 * (freq_len - grid_2)
    area = area_1 + area_2 + area_3
    area_max = torch.amax(area)
    area = torch.where(mask, area, area_max)
    split = torch.argmin(area)
    split_1 = (split // freq_len).item()
    split_2 = (split % freq_len).item()
    freq_1 = freq_grid_1.item()
    freq_2 = freq_s[split_1].item()
    freq_3 = freq_s[split_2].item()

    pos_seq = torch.arange(freq_len, device=device)
    pad_freqs = torch.cat(
        (
            freq_1 - freq_s[:split_1],
            freq_2 - freq_s[split_1:split_2],
            freq_3 - freq_s[split_2:freq_len],
        ),
        0,
    )
    num_1 = freq_1 * split_1
    num_2 = freq_2 * (split_2 - split_1)
    num_3 = freq_3 * (freq_len - split_2)
    pad_num = num_1 + num_2 + num_3 - index.shape[0]
    index_pad = torch.repeat_interleave(
        freq_idx_s[:freq_len], pad_freqs, 0, output_size=pad_num
    )
    index_w_pad = torch.cat((index, index_pad), 0)
    index_trans = torch.full(
        (block_num,), -1, device=device, dtype=pos_seq.dtype
    ).scatter_(0, freq_idx_s[:freq_len], pos_seq)
    new_index = index_trans[index_w_pad]
    num = num_1 + num_2 + num_3
    num_list = (num_1, num_2, num)
    freq_list = (freq_1, freq_2, freq_3)
    split_list = (split_1, split_2)

    return new_index, freq_idx_s, num_list, freq_list, split_list, freq_len


def pad_bmm_prepare(
    x_repeat: torch.Tensor,
    info: Tuple,
    list_info: Tuple,
    type: str,
):
    dtype = x_repeat.type
    device = x_repeat.device
    type, index, index_s, freq_idx_s, rank, freq_len, block_shape = info
    num_list, freq_list, split_list = list_info
    num_1, num_2, num = num_list
    freq_1, freq_2, freq_3 = freq_list
    split_1, split_2 = split_list

    index_len = index.shape[0]
    x_dim = x_repeat.shape[-1]
    y_dim = block_shape[-2] if type == "ffn" else block_shape[-1]
    len_dim_list = (index_len, x_dim, y_dim)

    index_s1 = index_s[:num_1]
    index_s2 = index_s[num_1 : num_1 + num_2]
    index_s3 = index_s[num_1 + num_2 :]
    index_s_list = (index_s1, index_s2, index_s3)

    freq_idx_s1 = freq_idx_s[:split_1]
    freq_idx_s2 = freq_idx_s[split_1:split_2]
    freq_idx_s3 = freq_idx_s[split_2:freq_len]
    freq_idx_s_list = (freq_idx_s1, freq_idx_s2, freq_idx_s3)

    if type == "forward":
        x_zone = torch.empty(num, x_dim, device=device, dtype=dtype)
    if type == "backward":
        x_zone = torch.zeros(num, x_dim, device=device, dtype=dtype)
    x_zone[rank] = x_repeat
    x1 = x_zone[:num_1].reshape(-1, freq_1, x_dim)
    x2 = x_zone[num_1 : num_1 + num_2].reshape(-1, freq_2, x_dim)
    x3 = x_zone[num_1 + num_2 :].reshape(-1, freq_3, x_dim)
    x_list = (x1, x2, x3)

    return len_dim_list, index_s_list, freq_idx_s_list, x_list


def pad_bmm_forward(
    x: torch.Tensor,
    block: torch.Tensor,
    bias: Optional[torch.Tensor],
    type: str,
    freq_idx_s: torch.Tensor,
):
    block = block[freq_idx_s]
    if type == "ffn":
        y1 = torch.bmm(x, block[:, 0])
        y2 = torch.bmm(x, block[:, 1])
        y3 = F.silu(y1) * y2
        y = torch.bmm(y3, block[:, 2].transpose(1, 2))
    else:
        y = torch.bmm(x, block)
        y1, y2 = None, None
    if bias.shape[0] > 1:
        y += bias[freq_idx_s]
    return y.flatten(0, 1), y1, y2


def pad_bmm_backward(
    xy_list: List, block: torch.Tensor, type: str, grad_block_option: bool
):
    grad_block = None
    if type == "ffn":
        x, y1, y1_silu, y2, y3, grad_y = xy_list
        grad_y3 = torch.bmm(grad_y, block[:, 2])
        if grad_block_option:
            grad_block_3 = torch.bmm(y3.transpose(1, 2), grad_y).transpose(1, 2)
        del y3, grad_y, xy_list[4], xy_list[5]
        grad_y1_silu = grad_y3 * y2
        grad_y2 = grad_y3 * y1_silu
        del grad_y3, y1_silu, y2, xy_list[2], xy_list[3]
        y1_sig = torch.sigmoid(y1)
        grad_y1 = grad_y1_silu * (y1_sig + y1 * y1_sig * (1 - y1_sig))
        del grad_y1_silu, y1_sig, y1, xy_list[1]
        if grad_block_option:
            grad_block_1 = torch.bmm(x.transpose(1, 2), grad_y1)
            grad_block_2 = torch.bmm(x.transpose(1, 2), grad_y2)
            grad_block = [grad_block_1, grad_block_2, grad_block_3]
        del x, xy_list[0]
        grad_x1 = torch.bmm(grad_y1, block[:, 0].transpose(1, 2))
        del grad_y1
        grad_x2 = torch.bmm(grad_y2, block[:, 1].transpose(1, 2))
        del grad_y2
        grad_x = grad_x1 + grad_x2
        del grad_x1, grad_x2
        return grad_x, grad_block
    else:
        x, grad_y = xy_list
        if grad_block_option:
            grad_block = torch.bmm(x.transpose(1, 2), grad_y)
        del x, xy_list[0]
        grad_x = torch.bmm(grad_y, block).transpose(1, 2)
        del grad_y, xy_list[1]
        return grad_x, grad_block
