import torch
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import json
import cv2
import os
from tqdm import tqdm, trange

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Cluster_Manager:
    def __init__(self , class_num = 0, cluster_config_file = None):
        self.class_num = class_num
        self.clusters = []
        if cluster_config_file is not None:
            self.load(cluster_config_file)
        return
    
    def load(self, cluster_config_file):
        with open(os.path.join(cluster_config_file, 'clusters.json'), 'r') as f:
            data = json.load(f)
        self.class_num = data['class_num']
        configs = data['cluster_dirs']
        assert self.class_num == len(configs)
        self.clusters = []
        for config in configs:
            if config is None:
                cluster = None
            else:
                cluster = Cluster(cluster_dir = config)
            self.clusters.append(cluster)
        print("load cluster num:",len(self.clusters))
    
    def save(self, cluster_manager_dir):
        os.makedirs(cluster_manager_dir, exist_ok=True)
        cluster_dirs = []
        cluster_manager_config_path = os.path.join(cluster_manager_dir, 'clusters.json')
        for i, cluster in enumerate(self.clusters):
            if cluster is None:
                cluster_dirs.append(None)
            else:
                cluster_dir = os.path.join(cluster_manager_dir, 'c'+str(i))
                cluster.save(cluster_dir)
                cluster_dirs.append(cluster_dir)
        manager_data = {'class_num':self.class_num, 'cluster_dirs':cluster_dirs}
        with open(cluster_manager_config_path, "w") as f:
            json.dump(manager_data, f)
            print('successfully save cluster manager to:',cluster_manager_config_path)
    
    def update_center(self,labels, pixels,quantile=0.3, n_samples=5000, band_factor = 0.5):
        print("updating clusers...")
        print(labels.shape, pixels.shape)
        idx = (labels==0)
        print(idx.shape, idx)
        idx = np.squeeze(idx)
        choose_pixel = pixels[idx]
        print(choose_pixel.shape)
        self.clusters = []
        for i in tqdm(range(self.class_num)):
            class_idx = np.squeeze(labels==i)
            class_pixels = pixels[class_idx]
            if len(class_pixels)==0:
                self.clusters.append(None)
                print('no pixels belong to class:',i)
                continue
            cluster = Cluster()
            cluster.update_center(class_pixels, quantile=quantile, n_samples=n_samples, band_factor = band_factor)
            self.clusters.append(cluster)
        return
    
    def dest_color(self,rgb,label):
        result = rgb.clone()
        for i in range(self.class_num):
            if self.clusters[i] is None:
                continue
            class_idx = torch.squeeze(label==i)
            class_rgb = rgb[class_idx]
            if class_rgb.shape[0] == 0:
                continue
            result[class_idx] = self.clusters[i].dest_color(class_rgb)
        return result
    
    def dest_class(self,rgb,label):
        result = torch.zeros([rgb.shape[0],1],dtype=torch.long).to(rgb.device)
        for i in range(self.class_num):
            if self.clusters[i] is None:
                continue
            class_idx = torch.squeeze(label==i)
            class_rgb = rgb[class_idx]
            if class_rgb.shape[0] == 0:
                continue
            result[class_idx] = self.clusters[i].dest_class(class_rgb)
        return result



class Cluster:
    def __init__(self, device = torch.device('cuda'), intensity_factor = 0.5, cluster_dir = None):
        self.batch_size = 10240 #
        self.anchors = None #
        self.links = None #
        self.rgb_centers = None
        self.device = device #
        self.intensity_factor = intensity_factor #
        if cluster_dir is not None:
            self.load(cluster_dir)
    
    def load(self, cluster_dir):
        with open(os.path.join(cluster_dir, 'config.json'), 'r') as f:
            data = json.load(f)
        self.batch_size = data['batch_size']
        self.intensity_factor = data['intensity_factor']
        self.anchors = torch.Tensor(data['anchors']).to(self.device)
        self.rgb_centers = torch.Tensor(data['rgb_centers']).to(self.device)
        self.links = torch.Tensor(data['links']).long().to(self.device)
        return
    
    def save(self, cluster_dir):
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_data = {"batch_size":self.batch_size, "intensity_factor":self.intensity_factor, "rgb_centers":self.rgb_centers.cpu().numpy().tolist(),\
            "anchors":self.anchors.cpu().numpy().tolist(), "links":self.links.cpu().numpy().tolist()}
        cluster_config_path = os.path.join(cluster_dir, 'config.json')
        with open(cluster_config_path, "w") as f:
            json.dump(cluster_data, f)
            print('successfully save cluster to:',cluster_config_path)
        
        for i in range(self.rgb_centers.shape[0]):
            color = self.rgb_centers[i]
            color_img = to8b(np.ones((50,50,3))*color.cpu().numpy())
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(cluster_dir, str(i)+'.png'), color_img)
        return

    def update_center(self,pixels, quantile=0.3, n_samples=5000, band_factor = 0.5): #对新采样的一组点进行聚类，更新类中心点
        pixels = self.mapping_color_np(pixels)
        bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=n_samples)
        bandwidth = max(bandwidth*band_factor,0.01)
        print('bandwidth:',bandwidth)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pixels)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters)
        self.choose_anchors(pixels, labels)
        centers = torch.from_numpy(ms.cluster_centers_).to(self.device) 
        self.rgb_centers = self.inv_mapping_color(centers)
        self.rgb_centers = self.rgb_centers.clamp(0,1)
    
    def choose_anchors(self,pixels,labels):
        pixels = torch.from_numpy(pixels).to(self.device) 
        labels = torch.from_numpy(labels).to(self.device) 
        print("before merge:",pixels.shape)
        print("choosing anchors...")
        N = pixels.shape[0]
        leaf_size = 0.01
        size_x = int(1/leaf_size)
        size_y = int(1/leaf_size)
        size_z = int(1/leaf_size)
        voxel = torch.zeros((size_x,size_y,size_z,3), dtype = torch.float32).to(self.device)
        voxel_label = torch.zeros((size_x,size_y,size_z,1), dtype=torch.long).to(self.device)-1
        half_leaf_size = leaf_size/2

        id = torch.clamp((pixels/leaf_size).long(),0, size_x-1)
        voxel_center = id*leaf_size+half_leaf_size
        dist = torch.sum((voxel_center - pixels)**2,dim=1)
        _, sorted_indices = torch.sort(dist, descending=True)
        id = id[sorted_indices,:]
        pixels = pixels[sorted_indices,:]
        labels = labels[sorted_indices,None]
        voxel[id[:,0], id[:,1],id[:,2]] = pixels
        voxel_label[id[:,0], id[:,1],id[:,2]] = labels
        valid_voxel_id = torch.squeeze(voxel_label>=0)
        self.anchors = voxel[valid_voxel_id]
        self.links = voxel_label[valid_voxel_id]
        print("after merge:", self.anchors.shape)
    
    def choose_anchors0(self,pixels,labels):
        print("before merge:",pixels.shape)
        print("choosing anchors...")
        N = pixels.shape[0]
        leaf_size = 0.01
        size_x = np.int(1/leaf_size)
        size_y = np.int(1/leaf_size)
        size_z = np.int(1/leaf_size)
        voxel = np.zeros(shape = (size_x,size_y,size_z,3), dtype = np.float32)
        voxel_near = np.zeros(shape = (size_x,size_y,size_z,1), dtype = np.float32)+1
        voxel_label = np.zeros(shape = (size_x,size_y,size_z,1), dtype=np.int32)-1
        half_leaf_size = leaf_size/2
        for i in range(N):
            id_x = np.clip(np.int(pixels[i,0]/leaf_size),0, size_x-1)
            id_y = np.clip(np.int(pixels[i,1]/leaf_size),0, size_y-1)
            id_z = np.clip(np.int(pixels[i,2]/leaf_size),0, size_z-1)
            voxel_center = np.array([id_x*leaf_size+half_leaf_size, id_y*leaf_size+half_leaf_size, id_z*leaf_size+half_leaf_size])
            dist = np.linalg.norm(voxel_center-pixels[i])
            if(dist < voxel_near[id_x,id_y,id_z]):
                voxel_near[id_x,id_y,id_z] = dist
                voxel[id_x,id_y,id_z] = pixels[i]
                voxel_label[id_x,id_y,id_z] = labels[i]

        voxel = voxel.reshape(-1,3)
        voxel_label = voxel_label.reshape(-1,1)
        self.anchors = []
        self.links = []
        for i in range(voxel.shape[0]):
            if voxel_label[i]==-1:
                continue
            self.anchors.append(voxel[i,None])
            self.links.append(voxel_label[i,None])
        self.anchors = np.concatenate(self.anchors, 0)
        self.links = np.concatenate(self.links, 0)
        print("after merge:", self.anchors.shape)
        self.anchors = torch.from_numpy(self.anchors).to(self.device)
        self.links = torch.from_numpy(self.links).long().to(self.device)
        
    
    def dest_color(self,rgb):
        d_rgb = self.mapping_color(rgb)
        start_idx = 0
        idxs = []
        while start_idx < d_rgb.shape[0]:
            end_idx = min(d_rgb.shape[0], start_idx+self.batch_size)
            idx = self.nearest_anchor(d_rgb[start_idx:end_idx])
            idxs.append(idx)
            start_idx = end_idx
        idxs = torch.cat(idxs, 0)
        return torch.squeeze(self.rgb_centers[self.links[idxs]])
    
    def dest_class(self,rgb):
        d_rgb = self.mapping_color(rgb)
        start_idx = 0
        idxs = []
        while start_idx < d_rgb.shape[0]:
            end_idx = min(d_rgb.shape[0], start_idx+self.batch_size)
            idx = self.nearest_anchor(d_rgb[start_idx:end_idx])
            idxs.append(idx)
            start_idx = end_idx
        idxs = torch.cat(idxs, 0)
        return self.links[idxs]
    
    def compute_dist(self, a,b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return sum_sq_a+sum_sq_b-2*a.mm(bt)
    
    def nearest_anchor(self,d_rgb):
        dist = self.compute_dist(self.anchors, d_rgb)
        idx = torch.argmin(dist, dim = 0).long()
        return idx
    
    
    def mapping_color0(self,rgb):
        return rgb
    
    def mapping_color_np(self, rgb):
        intensity = np.sum(rgb,axis=-1)
        d_rgb = np.zeros_like(rgb)
        d_rgb[...,0] = intensity/3.0*self.intensity_factor
        d_rgb[...,1] = rgb[...,1]/intensity
        d_rgb[...,2] = rgb[...,2]/intensity
        return d_rgb
    
    def mapping_color(self,rgb):
        intensity = torch.sum(rgb,axis=-1)
        d_rgb = torch.zeros_like(rgb).to(self.device)
        d_rgb[...,0] = intensity/3.0*self.intensity_factor
        d_rgb[...,1] = rgb[...,1]/intensity
        d_rgb[...,2] = rgb[...,2]/intensity
        return d_rgb
        
    def inv_mapping_color0(self,d_rgb):
        return d_rgb
    
    def inv_mapping_color(self,d_rgb):
        intensity = d_rgb[...,0]*3.0/self.intensity_factor
        rgb = torch.zeros_like(d_rgb).to(self.device)
        rgb[...,1] = d_rgb[...,1]*intensity
        rgb[...,2] = d_rgb[...,2]*intensity
        rgb[...,0] = intensity - rgb[...,1] - rgb[...,2]
        return rgb