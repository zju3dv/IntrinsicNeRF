import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import json
import cv2

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def compute_chroma_loss(color1, color2):
    sum1 = torch.sum(color1, axis=-1) + 1e-5
    r1 = color1[:,0]/sum1
    g1 = color1[:,1]/sum1
    sum2 = torch.sum(color2, axis=-1) + 1e-5
    r2 = color2[:,0]/sum2
    g2 = color2[:,1]/sum2
    return torch.mean((r1 - r2) ** 2) + torch.mean((g1 - g2) ** 2)

def compute_residual_loss(residual):
    return torch.mean(residual**2)

def compute_chroma_weight(color1, color2, obj_mask1, obj_mask2):
    sum1 = torch.sum(color1, axis=-1) + 1e-5
    r1 = color1[:,0]/sum1
    g1 = color1[:,1]/sum1
    sum2 = torch.sum(color2, axis=-1) + 1e-5
    r2 = color2[:,0]/sum2
    g2 = color2[:,1]/sum2
    weight = torch.exp(-60*((r1 - r2) ** 2 + (g1 - g2) ** 2)) * obj_mask1 * obj_mask2
    weight2 = ((r1 - r2) ** 2 + (g1 - g2) ** 2) * obj_mask1 * obj_mask2
    return weight, weight2

def compute_depth_weight(disp, disp2, acc, acc2):
    with torch.no_grad():
        mask = acc * acc2
        dist = torch.sqrt((disp-disp2)**2)
        dist = torch.where(torch.isnan(dist), torch.full_like(dist, 1), dist)
        mask *= torch.exp(-100*dist)
    return mask

def compute_reflect_sparsity_loss(albedo1, albedo2, w_chroma, w_depth):
    albedo_dist = albedo1-albedo2
    norm_2 = torch.sum(albedo_dist**2, axis=-1)
    return torch.mean(w_chroma * w_depth * norm_2)

def compute_shading_smooth_loss(shading1, shading2, w_inv_chroma, w_depth):
    return torch.mean(w_inv_chroma * w_depth * (shading1 - shading2)**2)

def compute_intensity_loss(gt_rgb, albedo):
    rgb_mean = torch.mean(gt_rgb)
    albedo_mean = torch.mean(albedo)
    return (rgb_mean-albedo_mean)**2

def compute_intrinsic_loss(albedo, shading, residual, gt_rgb, disp, acc, obj_mask):
    split = albedo.shape[0]//2
    albedo1 = albedo[:split]
    albedo2 = albedo[-1*split:]
    shading1 = shading[:split]
    shading2 = shading[-1*split:]
    gt_rgb1 = gt_rgb[:split]
    gt_rgb2 = gt_rgb[-1*split:]
    disp1 = disp[:split]
    disp2 = disp[-1*split:]
    acc1 = acc[:split]
    acc2 = acc[-1*split:]
    obj_mask1 = obj_mask[:split]
    obj_mask2 = obj_mask[-1*split:]

    intensity_loss = compute_intensity_loss(gt_rgb, albedo)
    residual_loss = compute_residual_loss(residual)
    chroma_loss = compute_chroma_loss(albedo, gt_rgb) 
    w_chroma, inv_w_chorma = compute_chroma_weight(gt_rgb1, gt_rgb2, obj_mask1, obj_mask2)
    w_depth = compute_depth_weight(disp1, disp2, acc1, acc2)
    reflect_sparsity_loss = compute_reflect_sparsity_loss(albedo1, albedo2, w_chroma, 1)
    shading_smooth_loss = compute_shading_smooth_loss(shading1, shading2, inv_w_chorma, 1) #b inv c:-1
    split2 = albedo1.shape[0]//2
    w_far_chroma, _ = compute_chroma_weight(gt_rgb1[:split2], gt_rgb1[-1*split2:], obj_mask1[:split2], obj_mask1[-1*split2:])
    w_far_depth = compute_depth_weight(disp1[:split2], disp1[-1*split2:], acc1[:split2], acc1[-1*split2:])
    far_reflect_loss = compute_reflect_sparsity_loss(albedo1[:split2], albedo1[-1*split2:], w_far_chroma, 1)
    
    return chroma_loss, residual_loss, reflect_sparsity_loss, shading_smooth_loss, far_reflect_loss, intensity_loss

'''记录
def color_mse(color1, color2):
    sum1 = torch.sum(color1, axis=-1) + 1e-5
    r1 = color1[:,0]/sum1
    g1 = color1[:,1]/sum1
    sum2 = torch.sum(color2, axis=-1) + 1e-5
    r2 = color2[:,0]/sum2
    g2 = color2[:,1]/sum2
    return torch.mean((r1 - r2) ** 2) + torch.mean((g1 - g2) ** 2)

def chrom_weight(color1, color2):
    sum1 = torch.sum(color1, axis=-1) + 1e-5
    mask1 = (sum1<3).double()
    r1 = color1[:,0]/sum1
    g1 = color1[:,1]/sum1
    sum2 = torch.sum(color2, axis=-1) + 1e-5
    mask2 = (sum2<3).double()
    r2 = color2[:,0]/sum2
    g2 = color2[:,1]/sum2
    result = torch.exp(-60*((r1 - r2) ** 2 + (g1 - g2) ** 2))
    result2 = ((r1 - r2) ** 2 + (g1 - g2) ** 2)
    #return torch.exp(-30*(torch.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2+1e-8)))#a
    #return torch.exp(-60*(torch.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2+1e-8)))#b
    #return torch.exp(-60*((r1 - r2) ** 2 + (g1 - g2) ** 2)) #c
    return mask1*mask2*result, mask1*mask2*result2#mask1*mask2*(1-result)
    #return torch.exp(-30*((r1 - r2) ** 2 + (g1 - g2) ** 2))

def far_chrom_weight(color):
    s = color.shape[0]//2
    color1 = color[0:s] 
    color2 = color[-s:]
    #print(color.shape, color1.shape, color2.shape)
    sum1 = torch.sum(color1, axis=-1) + 1e-5
    mask1 = (sum1<3).double()
    i1 = torch.mean(color1, axis=-1)/2
    r1 = color1[:,0]/sum1
    g1 = color1[:,1]/sum1
    sum2 = torch.sum(color2, axis=-1) + 1e-5
    mask2 = (sum2<3).double()
    i2 = torch.mean(color2, axis=-1)/2
    r2 = color2[:,0]/sum2
    g2 = color2[:,1]/sum2
    result = torch.exp(-30*(torch.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (i1 - i2) ** 2 +1e-8)))
    return mask1*mask2*result, mask1*mask2*(1-result)

def far_relate_cost(w_chrom_far, albedo1, albedo2):
    albedo_dist = albedo1-albedo2
    norm_2 = torch.sum(albedo_dist**2, axis=-1)
    return torch.mean(w_chrom*norm_2)

def compute_target_albedo_intensity(color):
    img_intensity = torch.mean(color, axis=-1)
    target_intensity = torch.zeros(color.shape[0])
    for i in range(color.shape[0]):
        w = chrom_weight(color,color[i].reshape(1,3))
        target_intensity[i] = torch.sum(w*img_intensity)/torch.sum(w)
    return target_intensity


def intensity_weight(color):
    w_i = 1
    rgb_i = color[:,0]*0.299+color[:,1]*0.587+color[:,2]*0.114
    return w_i*rgb_i + 1 - w_i

def shading_smooth_cost(mask, inv_w_chrom, shading1, shading2):
    return torch.mean(mask*inv_w_chrom*(torch.pow(shading1-shading2, 2)))


def reflectance_sparsity_cost(w_intensity, w_chrom, albedo1, albedo2):
    albedo_dist = albedo1-albedo2
    #print(albedo_dist)
    #norm_2 = torch.sum(albedo_dist*albedo_dist, axis=-1)
    norm_2 = torch.sqrt(torch.sum(albedo_dist*albedo_dist+1e-6, axis=-1))
    #print(norm_2)
    #return torch.mean(w_chrom*norm_2)
    return torch.mean(w_chrom*(torch.pow(norm_2, 0.8)))



def rsc(albedo1, albedo2, w_chrom, mask):
    albedo_dist = albedo1-albedo2
    norm_2 = torch.sum(albedo_dist**2, axis=-1)
    return torch.mean(mask*w_chrom*norm_2)

def albedo_intensity_cost(albedo, target_s):
    albedo_mean = torch.mean(albedo, axis=-1)
    return torch.mean((albedo_mean - target_s)**2)
    albedo_mean = torch.mean(albedo)
    target_s_mean = torch.mean(target_s)
    return (albedo_mean-target_s_mean)**2

def compute_all_loss(target_s,target_s2,target_a,target_a2,rgb,rgb2,albedo,albedo2,shading,shading2):
    
    return 1

def compute_valid_mask(disp, disp2, acc, acc2):
    with torch.no_grad():
        mask = acc * acc2
        dist = torch.sqrt((disp-disp2)**2)
        dist = torch.where(torch.isnan(dist), torch.full_like(dist, 0), dist)
        #print(torch.max(dist), torch.min(dist))
        mask *= torch.exp(-100*dist)
    return mask

'''

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.shading_linear = nn.Linear(W//2, 3)

            #albedo
            self.albedo_linear1 = nn.Linear(W, W//2)
            self.albedo_linear2 = nn.Linear(W//2, 3)

            self.test_linear1 = nn.Linear(W, W//2)
            self.test_linear2 = nn.Linear(W//2, 1)
            
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            #albedo
            albedo = self.albedo_linear1(h)
            albedo = F.relu(albedo)
            albedo = self.albedo_linear2(albedo)
            albedo = F.sigmoid(albedo)

            #shading
            shading = self.test_linear1(h)
            shading = F.relu(shading)
            shading = self.test_linear2(shading)
            shading = F.sigmoid(shading)

            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            residual = self.shading_linear(h)
            #residual = view_result
            residual = F.sigmoid(residual)
            #shading = view_result[:,[0]]
            #residual = view_result[:,1:4]
            #print(shading.shape, residual.shape)
            rgb  = albedo*shading + residual
            outputs = torch.cat([rgb, alpha, albedo, shading, residual], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
