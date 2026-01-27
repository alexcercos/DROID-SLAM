import torch
import torch.nn.functional as F

import droid_backends

import geom.projective_ops as pops
from lietorch import SE3

class DepthCorrBlock:
    def __init__(self, dmaps, poses, intrinsics, ii, jj, device, num_levels=4, radius=3):
        self.num_levels = num_levels

        Gs = SE3(poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, dmaps[None], intrinsics[None], ii, jj, return_depth=True)

        depth_jj = 1.0 / dmaps[jj]

        x = coords[..., 0]
        y = coords[..., 1]

        B, N, H, W = x.shape

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        wx = x - x0.float()
        wy = y - y0.float()

        w00 = (1-wx)*(1-wy)
        w01 = (1-wx)*wy
        w10 = wx*(1-wy)
        w11 = wx*wy

        in_bounds = (
            (x0 >= 0) & (x1 <= W) &
            (y0 >= 0) & (y1 <= H)
        ).unsqueeze(-1)

        dmask = valid_mask.bool() & in_bounds

        x0 = x0.clamp(0, W-1)
        y0 = y0.clamp(0, H-1)
        x1 = x1.clamp(0, W-1)
        y1 = y1.clamp(0, H-1)

        frame_idx = torch.arange(N, device=device)[None, :, None, None].expand(B, N, H, W)
        d00 = depth_jj[frame_idx, y0, x0]
        d01 = depth_jj[frame_idx, y1, x0]
        d10 = depth_jj[frame_idx, y0, x1]
        d11 = depth_jj[frame_idx, y1, x1]

        depth_sampled = w00*d00 + w01*d01 + w10*d10 + w11*d11

        zdepth = 1.0 / coords[..., 2]
        depth_error = (depth_sampled - zdepth).abs().unsqueeze(-1)
        
        K = 10.0
        corr = torch.exp(-K * depth_error) * dmask
        corr = corr.squeeze(-1) # Remove last dimension (1)

        self.corr_pyramid = []

        batch, num, ht, wd = corr.shape
        rd = radius*2+1

        for i in range(self.num_levels):
            patches = F.unfold(corr, kernel_size=rd, padding=radius)
            # patches: (B, num*rd*rd, H*W)

            sample = patches.view(batch, num, rd*rd, ht, wd)
            self.corr_pyramid.append(sample)
            corr = F.avg_pool2d(corr, 2, stride=2)
            batch, num, ht, wd = corr.shape
    
    def __call__(self, coords):
        out_pyramid = []

        ix = coords[..., 0].long()
        iy = coords[..., 1].long()

        for i in range(self.num_levels):
            batch, num, _, ht, wd = self.corr_pyramid[i].shape

            ix = ix.clamp(0, wd - 1)
            iy = iy.clamp(0, ht - 1)

            sample_out = self.corr_pyramid[i][
                torch.arange(batch)[:, None, None, None],
                torch.arange(num)[None, :, None, None],
                :,
                iy,
                ix,
            ]
            #indexing moves ":" to last dim
            sample_out = sample_out.permute(0, 1, 4, 2, 3)

            ix = torch.div(ix, 2, rounding_mode='trunc') #ix//2
            iy = torch.div(iy, 2, rounding_mode='trunc') #iy//2

            out_pyramid.append(sample_out)

        return torch.cat(out_pyramid, dim=2)

class CorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = droid_backends.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = droid_backends.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, num, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*num*h1*w1, 1, h2, w2)
        
        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch*num, h1, w1, h2//2**i, w2//2**i))
            corr = F.avg_pool2d(corr, 2, stride=2)
            
    def __call__(self, coords):
        out_pyramid = []
        batch, num, ht, wd, _ = coords.shape
        coords = coords.permute(0,1,4,2,3)
        coords = coords.contiguous().view(batch*num, 2, ht, wd)
        
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords/2**i, self.radius)
            out_pyramid.append(corr.view(batch, num, -1, ht, wd))

        return torch.cat(out_pyramid, dim=2)

    def cat(self, other):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)
        return self

    def __getitem__(self, index):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = self.corr_pyramid[i][index]
        return self


    @staticmethod
    def corr(fmap1, fmap2):
        """ all-pairs correlation """
        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 4.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 4.0
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, num, ht, wd, ht, wd)


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        ctx.r = r
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = droid_backends.altcorr_forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            droid_backends.altcorr_backward(fmap1, fmap2, coords, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None


class AltCorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius

        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B*N, C, H, W) / 4.0
        
        self.pyramid = []
        for i in range(self.num_levels):
            sz = (B, N, H//2**i, W//2**i, C)
            fmap_lvl = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(fmap_lvl.view(*sz))
            fmaps = F.avg_pool2d(fmaps, 2, stride=2)
  
    def corr_fn(self, coords, ii, jj):
        B, N, H, W, S, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3, 5)

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][:, ii]
            fmap2_i = self.pyramid[i][:, jj]

            coords_i = (coords / 2**i).reshape(B*N, S, H, W, 2).contiguous()
            fmap1_i = fmap1_i.reshape((B*N,) + fmap1_i.shape[2:])
            fmap2_i = fmap2_i.reshape((B*N,) + fmap2_i.shape[2:])

            corr = CorrLayer.apply(fmap1_i.float(), fmap2_i.float(), coords_i, self.radius)
            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=2)
        return corr


    def __call__(self, coords, ii, jj):
        squeeze_output = False
        if len(coords.shape) == 5:
            coords = coords.unsqueeze(dim=-2)
            squeeze_output = True

        corr = self.corr_fn(coords, ii, jj)
        
        if squeeze_output:
            corr = corr.squeeze(dim=-1)

        return corr.contiguous()

