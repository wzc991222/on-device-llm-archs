import torch
from torch.nn import functional as F
from torch.autograd import Function
from einops import rearrange, repeat
from grouped_gemm.backend import gmm  # type: ignore
from config import Config


def grouped_gemm_forward(
    x: torch.Tensor,
    block: torch.Tensor,
    index: torch.Tensor,
    config: Config,
    forward: bool = True,
):
    gmm_option = config.grouped_gemm_option == "gmm"
    for_loop_option = config.grouped_gemm_option == "for_loop"
    repeat_num = config.expert_num_per_token

    x_repeat = repeat(x, "b s d -> (b s n) d", n=repeat_num)
    index_s = torch.argsort(index, stable=False)
    index_freq = torch.bincount(index, minlength=block.shape[1])
    x_s = x_repeat[index_s]

    del x_repeat
    if gmm_option:
        index_freq = index_freq.to("cpu")
        x_s = x_s.to(torch.bfloat16)
    if for_loop_option:
        index_freq = index_freq.tolist()
    x_st = None
    if not forward and for_loop_option:
        x_st = torch.split(x_s.t(), index_freq, 1)
    if gmm_option:
        y_s1 = gmm(x_s, block[0], index_freq)
        y_s2 = gmm(x_s, block[1], index_freq)
        y_s1s = F.silu(y_s1)
        y_s3 = y_s1s * y_s2
        y_s = gmm(y_s3, block[2], index_freq, trans_b=True)
    if for_loop_option:
        x_s = torch.split(x_s, index_freq, 0)
        block_1 = torch.unbind(block[0], 0)
        block_2 = torch.unbind(block[1], 0)
        block_3t = torch.unbind(block[2].transpose(1, 2), 0)
        y_s1 = [x @ b for x, b in zip(x_s, block_1)]
        y_s2 = [x @ b for x, b in zip(x_s, block_2)]
        y_s1 = torch.cat(y_s1, 0)
        y_s2 = torch.cat(y_s2, 0)
        y_s1s = F.silu(y_s1)
        y_s3 = y_s1s * y_s2
        y_s3 = torch.split(y_s3, index_freq, 0)
        y_s = [y @ b for y, b in zip(y_s3, block_3t)]
        y_s = torch.cat(y_s, 0)

    if forward:
        return y_s, repeat_num, index_s
    else:
        x_list = x_s, x_st
        y_list = y_s1, y_s1s, y_s2, y_s3, y_s
        index_list = index_s, index_freq
        info_list = gmm_option, for_loop_option, repeat_num
        return x_list, y_list, index_list, info_list


def grouped_gemm_func_original(
    x: torch.Tensor,
    block: torch.Tensor,
    index: torch.Tensor,
    score: torch.Tensor,
    config: Config,
):
    y_s, repeat_num, index_s = grouped_gemm_forward(x, block, index, config)
    y_o = torch.empty_like(y_s)
    y_o[index_s] = y_s
    del y_s

    rms = None
    if config.expert_post_norm:
        y_sq = y_o.pow(2)
        y_sq_mean = y_sq.mean(dim=-1, keepdim=True)
        del y_sq
        rms = torch.sqrt(y_sq_mean + config.norm_eps)
        y = y_o / rms
    else:
        y = y_o

    y_output = y * score
    del y_o, y
    y_output = rearrange(
        y_output,
        "(b s n) d -> b s n d",
        s=config.seq_len,
        n=repeat_num,
    )
    y_output = torch.sum(y_output, 2, dtype=x.dtype)
    return y_output


class GroupedGEMM(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        block: torch.Tensor,
        index: torch.Tensor,
        score: torch.Tensor,
        config: Config,
    ):
        y_output = grouped_gemm_func_original(x, block, index, score, config)
        ctx.save_for_backward(x, block, index, score)
        ctx.info = config

        return y_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        config = ctx.info
        x, block, index, score = ctx.saved_tensors
        grad_block_option = ctx.needs_input_grad[1]
        x_list, y_list, index_list, info_list = grouped_gemm_forward(
            x, block, index, config, False
        )
        x_s, x_st = x_list
        y_s1, y_s1s, y_s2, y_s3, y_s = y_list
        index_s, index_freq = index_list
        gmm_option, for_loop_option, repeat_num = info_list
        del x_list, y_list, index_list, info_list
        grad_block = None

        grad_o = repeat(grad_output, "b s d -> (b s n) d", n=repeat_num)
        grad_o_s = grad_o[index_s]
        del grad_o
        grad_score_s = torch.sum(grad_o_s * y_s, -1, dtype=score.dtype, keepdim=True)
        grad_score = torch.empty_like(grad_score_s)
        grad_score[index_s] = grad_score_s
        score_s = score[index_s]
        grad_y_s_o = grad_o_s * score_s
        del grad_o_s

        if config.expert_post_norm:
            y_s_sq = y_s.pow(2)
            y_s_sq_mean = y_s_sq.mean(dim=-1, keepdim=True)
            del y_s_sq
            rms_s = torch.sqrt(y_s_sq_mean + config.norm_eps)
            y_s_rms_s = y_s / rms_s
            grad_y_s = grad_y_s_o / rms_s - (grad_y_s_o * y_s_rms_s).sum(
                dim=-1, keepdim=True
            ) * y_s_rms_s / (grad_y_s_o.shape[-1] * rms_s)
            del grad_y_s_o
        else:
            grad_y_s = grad_y_s_o

        if gmm_option:
            grad_y_s = grad_y_s.to(torch.bfloat16)
            grad_y_s3 = gmm(grad_y_s, block[2], index_freq)
            if grad_block_option:
                grad_block_3 = gmm(grad_y_s, y_s3, index_freq, trans_a=True)
        if for_loop_option:
            if grad_block_option:
                grad_y_st = torch.split(grad_y_s.t(), index_freq, 1)
                grad_block_3 = [y1 @ y2 for y1, y2 in zip(grad_y_st, y_s3)]
                grad_block_3 = torch.stack(grad_block_3, 0)
                del grad_y_st
            grad_y_s = torch.split(grad_y_s, index_freq, 0)
            block_3 = torch.unbind(block[2], 0)
            grad_y_s3 = [y @ b for y, b in zip(grad_y_s, block_3)]
            grad_y_s3 = torch.cat(grad_y_s3, 0)
        del grad_y_s, y_s3

        grad_y_s1s = grad_y_s3 * y_s2
        grad_y_s2 = grad_y_s3 * y_s1s
        del grad_y_s3, y_s2, y_s1s
        y_s1_sig = torch.sigmoid(y_s1)
        grad_y_s1 = grad_y_s1s * (y_s1_sig + y_s1 * y_s1_sig * (1 - y_s1_sig))
        del grad_y_s1s, y_s1, y_s1_sig

        if gmm_option:
            if grad_block_option:
                grad_block_1 = gmm(x_s, grad_y_s1, index_freq, trans_a=True)
                grad_block_2 = gmm(x_s, grad_y_s2, index_freq, trans_a=True)
            del x_s
            grad_x_s1 = gmm(grad_y_s1, block[0], index_freq, trans_b=True)
            grad_x_s2 = gmm(grad_y_s2, block[1], index_freq, trans_b=True)
        if for_loop_option:
            grad_y_s1 = torch.split(grad_y_s1, index_freq, 0)
            grad_y_s2 = torch.split(grad_y_s2, index_freq, 0)
            if grad_block_option:
                grad_block_1 = [x @ y for x, y in zip(x_st, grad_y_s1)]
                grad_block_2 = [x @ y for x, y in zip(x_st, grad_y_s2)]
                grad_block_1 = torch.stack(grad_block_1, 0)
                grad_block_2 = torch.stack(grad_block_2, 0)
            del x_s, x_st
            block_1t = torch.unbind(block[0].transpose(1, 2), 0)
            block_2t = torch.unbind(block[1].transpose(1, 2), 0)
            grad_x_s1 = [y @ b for y, b in zip(grad_y_s1, block_1t)]
            grad_x_s2 = [y @ b for y, b in zip(grad_y_s2, block_2t)]
            grad_x_s1 = torch.cat(grad_x_s1, 0)
            grad_x_s2 = torch.cat(grad_x_s2, 0)
        del grad_y_s1, grad_y_s2

        grad_x_s = grad_x_s1 + grad_x_s2
        del grad_x_s1, grad_x_s2
        if grad_block_option:
            grad_block = torch.stack((grad_block_1, grad_block_2, grad_block_3), 0)
            del grad_block_1, grad_block_2, grad_block_3

        grad_x = torch.empty_like(grad_x_s)
        grad_x[index_s] = grad_x_s
        del grad_x_s

        grad_x = rearrange(
            grad_x, "(b s n) d -> b s n d", s=config.seq_len, n=repeat_num
        )
        grad_x = torch.sum(grad_x, 2, dtype=x.dtype)
        if grad_block is not None:
            grad_block = grad_block.type_as(block)

        return grad_x, grad_block, None, grad_score, None
