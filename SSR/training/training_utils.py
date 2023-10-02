import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def batchify_rays(render_fn, rays_flat, chunk=1024 * 32):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_fn(rays_flat[i:i + chunk])
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def lr_poly_decay(base_lr, iter, max_iter, power):
    """ Polynomial learning rate decay
    Polynomial Decay provides a smoother decay using a polynomial function and reaches a learning rate of 0
    after max_update iterations.
    https://kiranscaria.github.io/general/2019/08/16/learning-rate-schedules.html

    max_iter: number of iterations to perform before the learning rate is taken to .
    power: the degree of the polynomial function. Smaller values of power produce slower decay and
        large values of learning rate for longer periods.
    """
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_exp_decay(base_lr, exp_base_lr, current_step, decay_steps):
    """ lr = lr0 * decay_base^(−kt)
    """
    new_lrate = base_lr * (exp_base_lr ** (current_step / decay_steps))
    return new_lrate


def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()


def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_label):
    if (true_labels == ignore_label).all():
        return [0]*4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids] 
    true_labels = true_labels[valid_pix_ids]
    
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious


def calculate_depth_metrics(depth_trgt, depth_pred):
    """ Computes 2d metrics between two depth maps
    
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth
    Returns:
        Dict of metrics
    """
    mask1 = depth_pred>0 # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt<10) * (depth_trgt>0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred-depth_trgt)
    abs_rel = abs_diff/depth_trgt
    sq_diff = abs_diff**2
    sq_rel = sq_diff/depth_trgt
    sq_log_diff = (np.log(depth_pred)-np.log(depth_trgt))**2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25**2).astype('float')
    r3 = (thresh < 1.25**3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics

img2mse = lambda x, y: torch.mean((x - y) ** 2)

#! 约束albedo色度与图像GT的色度之间差距不能过大
def compute_chroma_loss(color1, color2):
    sum1 = torch.sum(color1, axis=-1) + 1e-5
    r1 = color1[:,0]/sum1
    g1 = color1[:,1]/sum1
    sum2 = torch.sum(color2, axis=-1) + 1e-5
    r2 = color2[:,0]/sum2
    g2 = color2[:,1]/sum2
    return torch.mean((r1 - r2) ** 2) + torch.mean((g1 - g2) ** 2)

#! 约束残差项尽可能接近0，在前半段该约束权重较强，后半段减弱但依旧保持该约束
def compute_residual_loss(residual):
    return torch.mean(residual**2)

#! 通过两个像素点之间的颜色差异计算albedo以及shading损失的权重，Label项目用于将该约束限制在同一个类别的物体中
def compute_chroma_weight(color1, color2, label1, label2):
    sum1 = torch.sum(color1, axis=-1) + 1e-5
    r1 = color1[:,0]/sum1
    g1 = color1[:,1]/sum1
    sum2 = torch.sum(color2, axis=-1) + 1e-5
    r2 = color2[:,0]/sum2
    g2 = color2[:,1]/sum2
    mask = (label1==label2).float() #相同标签的掩模
    weight = torch.exp(-60*((r1 - r2) ** 2 + (g1 - g2) ** 2)) * mask # 这个权重用于albedo，差别越大权重越小
    weight2 = ((r1 - r2) ** 2 + (g1 - g2) ** 2) #* obj_mask1 * obj_mask2  # 这个权重用于相邻点shading，差别越大权重越大
    return weight, weight2

#! 计算深度权重，没有使用
def compute_depth_weight(disp, disp2, acc, acc2):
    with torch.no_grad():
        mask = acc * acc2
        dist = torch.sqrt((disp-disp2)**2)
        dist = torch.where(torch.isnan(dist), torch.full_like(dist, 1), dist)
        mask *= torch.exp(-100*dist)
    return mask

#! 计算两点之间albedo的约束
def compute_reflect_sparsity_loss(albedo1, albedo2, w_chroma, w_depth):
    albedo_dist = albedo1-albedo2
    norm_2 = torch.sum(albedo_dist**2, axis=-1)
    return torch.mean(w_chroma * w_depth * norm_2)

#! 计算相邻两点shading的约束
def compute_shading_smooth_loss(shading1, shading2, w_inv_chroma, w_depth):
    return torch.mean(w_inv_chroma * w_depth * (shading1 - shading2)**2)

#! 对一组点整体intensity的约束
def compute_intensity_loss(gt_rgb, albedo):
    rgb_mean = torch.mean(gt_rgb)
    albedo_mean = torch.mean(albedo)
    return (rgb_mean-albedo_mean)**2

#! 计算所有本征分解相关的损失
def compute_intrinsic_loss(albedo, shading, residual, gt_rgb, disp, acc, semantic_label):
    split = albedo.shape[0]//2 #! 用于将一组采样点分为前后两半
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
    semantic_label1 = semantic_label[:split]
    semantic_label2 = semantic_label[-1*split:]

    intensity_loss = compute_intensity_loss(gt_rgb, albedo)
    residual_loss = compute_residual_loss(residual)
    chroma_loss = compute_chroma_loss(albedo, gt_rgb) 
    w_chroma, inv_w_chorma = compute_chroma_weight(gt_rgb1, gt_rgb2, semantic_label1, semantic_label2)
    w_depth = compute_depth_weight(disp1, disp2, acc1, acc2)
    reflect_sparsity_loss = compute_reflect_sparsity_loss(albedo1, albedo2, w_chroma, 1) #a
    shading_smooth_loss = compute_shading_smooth_loss(shading1, shading2, inv_w_chorma, 1) #b inv c:-1
    split2 = albedo1.shape[0]//2
    #! 远距离的点是通过将前0~1/4 和 1/4~2/4这两段点进行对比
    w_far_chroma, _ = compute_chroma_weight(gt_rgb1[:split2], gt_rgb1[-1*split2:], semantic_label1[:split2], semantic_label1[-1*split2:])
    w_far_depth = compute_depth_weight(disp1[:split2], disp1[-1*split2:], acc1[:split2], acc1[-1*split2:])
    far_reflect_loss = compute_reflect_sparsity_loss(albedo1[:split2], albedo1[-1*split2:], w_far_chroma, 1)
    
    return chroma_loss, residual_loss, reflect_sparsity_loss, shading_smooth_loss, far_reflect_loss, intensity_loss
