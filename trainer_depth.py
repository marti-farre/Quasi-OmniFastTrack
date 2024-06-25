import glob
import os
import pdb
import time

import cv2
import imageio
from PIL import Image
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from criterion_depth import masked_mse_loss, masked_l1_loss, compute_depth_range_loss, lossfun_distortion, compute_depth_loss, compute_point_loss
from networks.mfn import GaborNet
from networks.nvp_simplified import NVPSimplified
from kornia import morphology as morph


torch.manual_seed(1234)


def init_weights(m):
    # Initializes weights according to the DCGAN paper
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class BaseTrainer():
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.seq_dir = args.data_dir

        self.read_data()

        self.feature_mlp = GaborNet(in_size=1,
                                    hidden_size=256,
                                    n_layers=2,
                                    alpha=4.5,
                                    out_size=128).to(device)

        self.deform_mlp = NVPSimplified(n_layers=6,
                                        feature_dims=128,
                                        hidden_size=[256, 256, 256],
                                        proj_dims=256,
                                        code_proj_hidden_size=[],
                                        proj_type='fixed_positional_encoding',
                                        pe_freq=args.pe_freq,
                                        normalization=False,
                                        affine=args.use_affine,
                                        device=device).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.feature_mlp.parameters(), 'lr': args.lr_feature},
            {'params': self.deform_mlp.parameters(), 'lr': args.lr_deform},
            {'params': self.depths, 'lr' : args.lr_depth}
        ])

        self.learnable_params = list(self.feature_mlp.parameters()) + \
                                list(self.deform_mlp.parameters())

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)
        seq_name = os.path.basename(args.data_dir.rstrip('/'))
        self.out_dir = os.path.join(args.save_dir, 'depth_{}_full'.format(seq_name))
        self.step = self.load_from_ckpt(self.out_dir,
                                        load_opt=self.args.load_opt,
                                        load_scheduler=self.args.load_scheduler)
        self.time_steps = torch.linspace(1, self.num_imgs, self.num_imgs, device=self.device)[:, None] / self.num_imgs

        if args.distributed:
            self.feature_mlp = torch.nn.parallel.DistributedDataParallel(
                self.feature_mlp,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
            self.deform_mlp = torch.nn.parallel.DistributedDataParallel(
                self.deform_mlp,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

    def read_data(self):
        self.seq_dir = self.args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')

        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        self.num_imgs = min(self.args.num_imgs, len(img_files))
        self.img_files = img_files[:self.num_imgs]

        images = np.array([imageio.imread(img_file) / 255. for img_file in self.img_files])
        self.images = torch.from_numpy(images).float()  # [n_imgs, h, w, 3]
        self.h, self.w = self.images.shape[1:3]

        mask_files = [img_file.replace('color', 'mask').replace('.jpg', '.png') for img_file in self.img_files]
        if os.path.exists(mask_files[0]):
            masks = np.array([imageio.imread(mask_file)[..., :3].sum(axis=-1) / 255.
                              if imageio.imread(mask_file).ndim == 3 else
                              imageio.imread(mask_file) / 255.
                              for mask_file in mask_files])
            self.masks = torch.from_numpy(masks).to(self.device) > 0.  # [n_imgs, h, w]
            self.with_mask = True
        else:
            self.masks = torch.ones(self.images.shape[:-1]).to(self.device) > 0.
            self.with_mask = False
        self.grid = util.gen_grid(self.h, self.w, device=self.device, normalize=False, homogeneous=True).float()

        self.depth_dir = os.path.join(self.seq_dir, 'depths') # UPDATE

        def load_depth_image(path):

            image = imageio.imread(path) / 255.  # Assuming grayscale depths
            image = image[:,:, 1]
            image = torch.from_numpy(image).float().unsqueeze(2)  # Normalize to [0, 1]
            return image # .unsqueeze(0)

        depths_init = torch.empty(self.num_imgs, self.h, self.w, 1)
        image_index = 0
        for filename in os.listdir(self.depth_dir):
            # Check if it's an image file (adjust based on your file format)
            if filename.endswith(".png"):
                # Get the full path
                image_path = os.path.join(self.depth_dir, filename)
                # Load the depth image
                depth_image = load_depth_image(image_path)
                depths_init[image_index] = depth_image  # Modify index for multiple images
                image_index += 1
                
        self.depths_init = depths_init.to(self.device)
        self.depths = self.depths_init.clone().detach().requires_grad_(True).to(self.device)


    def project(self, x, return_depth=False):
        '''
        orthographic projection
        :param x: [..., 3]
        :param return_depth: if returning depth
        :return: pixel_coords in image space [..., 2], depth [..., 1]
        '''
        pixel_coords, depth = torch.split(x, dim=-1, split_size_or_sections=[2, 1])
        pixel_coords = util.denormalize_coords(pixel_coords, self.h, self.w)
        if return_depth:
            return pixel_coords, depth
        else:
            return pixel_coords

    def unproject(self, pixels, depths):
        '''
        orthographic unprojection
        :param pixels: [..., 2] pixel coordinates (unnormalized)
        :param depths: [..., 1]
        :return: 3d locations in normalized space [..., 3]
        '''
        assert pixels.shape[-1] in [2, 3]
        assert pixels.ndim == depths.ndim
        pixels = util.normalize_coords(pixels[..., :2], self.h, self.w)
        return torch.cat([pixels, depths], dim=-1)

    def get_in_range_mask(self, x, max_padding=0):
        mask = (x[..., 0] >= -max_padding) * \
               (x[..., 0] <= self.w - 1 + max_padding) * \
               (x[..., 1] >= -max_padding) * \
               (x[..., 1] <= self.h - 1 + max_padding)
        return mask
    
    def get_prediction_one_way(self, x, id, inverse=False):
        '''
        mapping 3d points from local to canonical or from canonical to local (inverse=True)
        :param x: [n_imgs, n_pts, n_samples, 3]
        :param id: [n_imgs, ]
        :param inverse: True or False
        :return: [n_imgs, n_pts, n_samples, 3]
        '''
        t = self.time_steps[id]  # [n_imgs, 1]
        feature = self.feature_mlp(t)  # [n_imgs, n_feat]

        if inverse:
            if self.args.distributed:
                out = self.deform_mlp.module.inverse(t, feature, x)
            else:
                out = self.deform_mlp.inverse(t, feature, x)
        else:
            out = self.deform_mlp.forward(t, feature, x)

        return out  # [n_imgs, n_pts, n_samples, 3]

    def get_predictions(self, x1, id1, id2, return_canonical=False):
        '''
        mapping 3d points from one frame to another frame
        :param x1: [n_imgs, n_pts, n_samples, 3]
        :param id1: [n_imgs,]
        :param id2: [n_imgs,]
        :return: [n_imgs, n_pts, n_samples, 3]
        '''   
        x1_canonical = self.get_prediction_one_way(x1, id1)
        x2_pred = self.get_prediction_one_way(x1_canonical, id2, inverse=True)
        if return_canonical:
            return x2_pred, x1_canonical
        else:
            return x2_pred  # [n_imgs, n_pts, n_samples, 3]
    def gradient_loss(self, pred, gt, weight=None):
        '''
        coordinate
        :param pred: [n_imgs, n_pts, 2] or [n_pts, 2]
        :param gt:
        :return:
        '''
        pred_grad = pred[..., 1:, :] - pred[..., :-1, :]
        gt_grad = gt[..., 1:, :] - gt[..., :-1, :]
        if weight is not None:
            weight_grad = weight[..., 1:, :] * weight[..., :-1, :]
        else:
            weight_grad = None
        loss = masked_l1_loss(pred_grad, gt_grad, weight_grad)
        return loss
    
    def compute_scene_flow_smoothness_loss(self, ids, xs):
        mask_valid = (ids >= 1) * (ids < self.num_imgs - 1)
        ids = ids[mask_valid]
        if len(ids) == 0:
            return torch.tensor(0.)
        xs = xs[mask_valid]
        ids_prev = ids - 1
        ids_after = ids + 1
        xs_prev_after = self.get_predictions(torch.cat([xs, xs], dim=0),
                                             np.concatenate([ids, ids]),
                                             np.concatenate([ids_prev, ids_after]))
        xs_prev, xs_after = torch.split(xs_prev_after, split_size_or_sections=len(xs), dim=0)
        scene_flow_prev = xs - xs_prev
        scene_flow_after = xs_after - xs
        loss = masked_l1_loss(scene_flow_prev, scene_flow_after)
        return loss
    
    def canonical_sphere_loss(self, xs_canonical_samples, radius=1.):
        ''' encourage mapped locations to be within a (unit) sphere '''
        xs_canonical_norm = (xs_canonical_samples ** 2).sum(dim=-1)
        if (xs_canonical_norm >= 1.).any():
            canonical_unit_sphere_loss = ((xs_canonical_norm[xs_canonical_norm >= radius] - 1) ** 2).mean()
        else:
            canonical_unit_sphere_loss = torch.tensor(0.)
        return canonical_unit_sphere_loss
    
    def spatial_gradient(self, depth):
        grad_x = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        grad_y = depth[:, 1:, :, :] - depth[:, :-1, :, :]
        return grad_x, grad_y

    def compute_depth_optimization_loss(self, depth, depth_init):

        grad_depth_x, grad_depth_y = self.spatial_gradient(depth)
        grad_depth_init_x, grad_depth_init_y = self.spatial_gradient(depth_init)

        l2_loss_x = F.mse_loss(grad_depth_x, grad_depth_init_x)
        l2_loss_y = F.mse_loss(grad_depth_y, grad_depth_init_y)
        l2_loss = l2_loss_x + l2_loss_y

        l1_loss = F.l1_loss(depth, depth_init)

        loss = l2_loss + l1_loss
        return loss

    def compute_all_losses(self,
                           batch,
                           w_depth_range=10,
                           w_depth = 5.,
                           w_depth_reg = 1,
                           w_scene_flow_smooth=10.,
                           w_canonical_unit_sphere=0.,
                           write_logs=True,
                           return_data=False,
                           log_prefix='loss',
                           ):

        depth_min_th = self.args.min_depth
        depth_max_th = self.args.max_depth
        max_padding = self.args.max_padding

        ids1 = batch['ids1'].numpy()
        ids2 = batch['ids2'].numpy()
        px1s = batch['pts1'].to(self.device)
        px2s = batch['pts2'].to(self.device)
        # weights = batch['weights'].to(self.device)
        num_pts = px1s.shape[1]
        select_ids = batch['select_ids'].to(self.device)
        batch_mask = batch['batch_mask'].to(self.device)
        depth1 = self.depths[ids1][batch_mask][select_ids]
        depth2 = self.depths[ids2][batch_mask][select_ids]
        # [n_pair, n_pts, n_samples, 3]
        x1s_depth = self.unproject(px1s, depth1) # MODIFICACIÓ: x a partir de la nova depth
        x1s_depth = x1s_depth.unsqueeze(2)
        
        # x2s_proj_samples, x1s_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)
        x2s_proj_samples_depth, x1s_canonical_samples_depth = self.get_predictions(x1s_depth, ids1, ids2, return_canonical=True) # MODIFICACIÓ
        x2s_proj_samples_depth = x2s_proj_samples_depth.squeeze(dim=2)
        
        px2s_proj_depth, px2s_proj_depths_depth = self.project(x2s_proj_samples_depth, return_depth=True) # MODIFICACIÓ

        mask = self.get_in_range_mask(px2s_proj_depth, max_padding)
        
        
        point_loss = compute_point_loss(px2s_proj_depth, px2s, num_pts)

        # depth_reg_loss = self.compute_depth_optimization_loss(depth2, depth_init2)
        depth_reg_loss = self.compute_depth_optimization_loss(self.depths[ids2], self.depths_init[ids2])
        
        # the depth of mapped 3d locations should aproximate to actual depth
        depth_loss = compute_depth_loss(px2s_proj_depths_depth, depth2, num_pts)
                
        # mapped depth should be within the predefined range
        depth_range_loss = compute_depth_range_loss(px2s_proj_depths_depth, depth_min_th, depth_max_th)

        # scene flow smoothness
        # only apply to 25% of points to reduce cost
        scene_flow_smoothness_loss = self.compute_scene_flow_smoothness_loss(ids1, x1s_depth[:, :int(num_pts / 4)])

        # loss for mapped points to stay within canonical sphere
        canonical_unit_sphere_loss = self.canonical_sphere_loss(x1s_canonical_samples_depth)
        loss = point_loss + \
               w_depth * depth_loss + \
               w_depth_reg*depth_reg_loss + \
               w_depth_range * depth_range_loss + \
               w_scene_flow_smooth * scene_flow_smoothness_loss + \
               w_canonical_unit_sphere * canonical_unit_sphere_loss
        

        if write_logs:
            self.scalars_to_log['{}/Loss'.format(log_prefix)] = loss.item()
            self.scalars_to_log['{}/point_loss'.format(log_prefix)] = point_loss.item()
            self.scalars_to_log['{}/loss_depth'.format(log_prefix)] = depth_loss.item()
            self.scalars_to_log['{}/loss_depth_reg'.format(log_prefix)] = depth_reg_loss.item()
            self.scalars_to_log['{}/loss_depth_range'.format(log_prefix)] = depth_range_loss.item()
            self.scalars_to_log['{}/loss_scene_flow_smoothness'.format(log_prefix)] = scene_flow_smoothness_loss.item()
            self.scalars_to_log['{}/loss_canonical_unit_sphere'.format(log_prefix)] = canonical_unit_sphere_loss.item()
        data = {'ids1': ids1,
                'ids2': ids2,
                'depth1': depth1,
                'x1s': x1s_depth,
                'x2s_pred': x2s_proj_samples_depth,
                'xs_canonical': x1s_canonical_samples_depth,
                'mask': mask,
                'px2s_proj': px2s_proj_depth,
                'px2s_proj_depths': px2s_proj_depths_depth
                }
        if return_data:
            return loss, data
        else:
            return loss

    def weight_scheduler(self, step, start_step, w, min_weight, max_weight):
        if step <= start_step:
            weight = 0.0
        else:
            weight = w * (step - start_step)
        weight = np.clip(weight, a_min=min_weight, a_max=max_weight)
        return weight

    def train_one_step(self, step, batch):
        self.step = step
        start = time.time()
        self.scalars_to_log = {}

        self.optimizer.zero_grad()
        w_scene_flow_smooth = 20.

        loss, flow_data = self.compute_all_losses(batch,
                                                  w_scene_flow_smooth=w_scene_flow_smooth,
                                                  return_data=True)

        if torch.isnan(loss):
            pdb.set_trace()

        loss.backward()

        is_break = False
        for p in self.deform_mlp.parameters():
            if torch.isnan(p.data).any() or torch.isnan(p.grad).any():
                is_break = True

        for p in self.feature_mlp.parameters():
            if torch.isnan(p.data).any() or torch.isnan(p.grad).any():
                is_break = True

        if is_break:
            pdb.set_trace()

        if self.args.grad_clip > 0:
            for param in self.learnable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(param, self.args.grad_clip)
                if grad_norm > self.args.grad_clip:
                    print("Warning! Clip gradient from {} to {}".format(grad_norm, self.args.grad_clip))

        self.optimizer.step()
        self.scheduler.step()

        self.scalars_to_log['lr'] = self.optimizer.param_groups[0]['lr']

        self.scalars_to_log['time'] = time.time() - start
        self.ids1 = flow_data['ids1']
        self.ids2 = flow_data['ids2']
        self.depth1 = flow_data['depth1']

    def get_correspondences_for_pixels(self, ids1, px1s, depth, ids2,
                                       return_depth=False,
                                       use_max_loc=False):
        '''
        get correspondences for pixels in one frame to another frame
        :param ids1: [num_imgs]
        :param px1s: [num_imgs, num_pts, 2]
        :param ids2: [num_imgs]
        :param return_depth: if returning the depth of the mapped point in the target frame
        :param use_max_loc: if using only the sample with the maximum blending weight to
                            compute the corresponding location rather than doing over composition.
                            set to True leads to better results on occlusion boundaries,
                            by default it is set to True for inference.

        :return: px2s_pred: [num_imgs, num_pts, 2], and optionally depth: [num_imgs, num_pts, 1]
        '''
        # [n_pair, n_pts, n_samples, 3]
        x1s_depth = self.unproject(px1s, depth) # MODIFICACIÓ: x a partir de la nova depth
        x1s_depth = x1s_depth.unsqueeze(2)
        x2s_proj = self.get_predictions(x1s_depth, ids1, ids2, return_canonical=False)
        x2s_proj = x2s_proj.squeeze(dim=2)
        
        return self.project(x2s_proj, return_depth=return_depth) # MODIFICACIÓ

    def get_pred_flows(self, ids1, ids2, depths, chunk_size=40000, use_max_loc=False, return_original=False):
        grid = self.grid[..., :2].reshape(-1, 2)
        flows = []
        depths = torch.flatten(depths, start_dim = 1, end_dim = 2)
        for id1, id2, depth_grid in zip(ids1, ids2, torch.split(depths, split_size_or_sections=1, dim=0)):
            flow_map = []
            for coords, depth in zip(torch.split(grid, split_size_or_sections=chunk_size, dim=0), torch.split(depth_grid, split_size_or_sections=chunk_size, dim=1)):
                with torch.no_grad():
                    flows_chunk = self.get_correspondences_for_pixels([id1], coords[None], depth, [id2],
                                                                      use_max_loc=use_max_loc)[0]
                    flow_map.append(flows_chunk)
            flow_map = torch.cat(flow_map, dim=0).reshape(self.h, self.w, 2)
            flow_map = (flow_map - self.grid[..., :2]).cpu().numpy()
            flows.append(flow_map)
        flows = np.stack(flows, axis=0)
        flow_imgs = util.flow_to_image(flows)
        if return_original:
            return flow_imgs, flows
        else:
            return flow_imgs  # [n, h, w, 3], numpy array    
            
    def get_pred_depths_for_pixels(self, ids, depth, pixels):
        '''
        :param ids: list [n_imgs,]
        :param pixels: [n_imgs, n_pts, 2]
        :return: pred_depths: [n_imgs, n_pts, 1]
        '''
        # [n_pair, n_pts, n_samples, 3]
        x1s_depth = self.unproject(pixels, depth) # MODIFICACIÓ: x a partir de la nova depth
        x1s_depth = x1s_depth.unsqueeze(2)
        x2s_proj = self.get_predictions(x1s_depth, ids[0], ids[1], return_canonical=False)
        x2s_proj = x2s_proj.squeeze(dim=2)
        px2s, pred_depths = self.project(x2s_proj, return_depth=True) # MODIFICACIÓ
        return pred_depths  # [n_imgs, n_pts, 1]
    
    def sample_pts_within_mask(self, mask, num_pts, return_normed=False, seed=None,
                               use_mask=False, reverse_mask=False, regular=False, interval=10):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        if use_mask:
            if reverse_mask:
                mask = ~mask
            kernel = torch.ones(7, 7, device=self.device)
            mask = morph.erosion(mask.float()[None, None], kernel).bool().squeeze()  # Erosion
        else:
            mask = torch.ones_like(self.grid[..., 0], dtype=torch.bool)

        if regular:
            coords = self.grid[::interval, ::interval, :2][mask[::interval, ::interval]]
        else:
            coords_valid = self.grid[mask][..., :2]
            rand_inds = rng.choice(len(coords_valid), num_pts, replace=(num_pts > len(coords_valid)))
            coords = coords_valid[rand_inds]

        coords_normed = util.normalize_coords(coords, self.h, self.w)
        if return_normed:
            return coords, coords_normed
        else:
            return coords  # [num_pts, 2]
        
    def plot_correspondences_for_pixels(self, query_kpt, query_id, num_pts=200,
                                        vis_occlusion=False,
                                        occlusion_th=0.95,
                                        use_max_loc=False,
                                        radius=2,
                                        return_kpts=False):
        frames = []
        kpts = []
        with torch.no_grad():
            img_query = self.images[query_id].cpu().numpy()
            for id in range(0, self.num_imgs):
                if id == query_id:
                    kp_i = query_kpt
                else:
                    coords = query_kpt[None].squeeze(0)
                    y_coords = coords[:, 0].long()
                    x_coords = coords[:, 1].long()
                    depth = self.depths[query_id]
                    depth = depth[x_coords, y_coords, 0].unsqueeze(1)
                    depth = depth.unsqueeze(0)
                    kp_i = self.get_correspondences_for_pixels([query_id], query_kpt[None], depth, [id],
                                                                   use_max_loc=use_max_loc)[0]
                mask = None
                img_i = self.images[id].cpu().numpy()
                out = util.drawMatches(img_query, img_i, query_kpt.cpu().numpy(), kp_i.cpu().numpy(),
                                       num_vis=num_pts, mask=mask, radius=radius)
                frames.append(out)
                kpts.append(kp_i)
        kpts = torch.stack(kpts, dim=0)
        if return_kpts:
            return frames, kpts
        return frames
    
    def eval_video_correspondences(self, query_id, pts=None, num_pts=200, seed=1234, use_mask=False,
                                   mask=None, reverse_mask=False, vis_occlusion=False, occlusion_th=0.99,
                                   use_max_loc=False, regular=True,
                                   interval=10, radius=2, return_kpts=False):
        with torch.no_grad():
            if mask is not None:
                mask = torch.from_numpy(mask).bool().to(self.device)
            else:
                mask = self.masks[query_id]

            if pts is None:
                x_0 = self.sample_pts_within_mask(mask, num_pts, seed=seed, use_mask=use_mask,
                                                  reverse_mask=reverse_mask, regular=regular, interval=interval)
                num_pts = 1e7 if regular else num_pts
            else:
                x_0 = torch.from_numpy(pts).float().to(self.device)
            return self.plot_correspondences_for_pixels(x_0, query_id, num_pts=num_pts,
                                                        vis_occlusion=False,
                                                        occlusion_th=occlusion_th,
                                                        use_max_loc=use_max_loc,
                                                        radius=radius, return_kpts=return_kpts)

    def log(self, writer, step):
        if self.args.local_rank == 0:
            if step % self.args.i_print == 0:            
                logstr = '{}_{} | step: {} |'.format(self.args.expname, self.seq_name, step)
                for k in self.scalars_to_log.keys():
                    logstr += ' {}: {:.6f}'.format(k, self.scalars_to_log[k])
                    if k != 'time':
                        writer.add_scalar(k, self.scalars_to_log[k], step)
                print(logstr)
            
            if step % self.args.i_img == 0:

                # flow
                flows = self.get_pred_flows(self.ids1[0:1], self.ids2[0:1], self.depths[self.ids1[0:1]], chunk_size=self.args.chunk_size)[0] 
                writer.add_image('flow', flows, step, dataformats='HWC')

            if step % self.args.i_weight == 0 and step > 0:
                # save checkpoints
                os.makedirs(self.out_dir, exist_ok=True)
                print('Saving checkpoints at {} to {}...'.format(step, self.out_dir))
                fpath = os.path.join(self.out_dir, 'model_{:06d}.pth'.format(step))
                self.save_model(fpath)

                depth_save_dir = os.path.join(self.out_dir, 'optimized_depths')
                os.makedirs(depth_save_dir, exist_ok=True)
                print('saving optimized depths to {}...'.format(depth_save_dir))
                
                for i in range(self.depths.shape[0]):
                    tensor = self.depths[i, :, :, 0]
                    tensor_255 = (tensor * 255).byte()
                    numpy_array = tensor_255.cpu().numpy()
                    image = Image.fromarray(numpy_array)
                    if i < 10:
                        id = f"0000{i}"
                    else:
                        id = f"000{i}"
                    fpath = os.path.join(depth_save_dir, 'depth_opti_{}.png'.format(id))
                    image.save(fpath)

                vis_dir = os.path.join(self.out_dir, 'vis')
                os.makedirs(vis_dir, exist_ok=True)
                print('saving visualizations to {}...'.format(vis_dir))

                if self.with_mask:
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            use_mask=True,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_foreground_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                    
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            use_mask=True,
                                                                            reverse_mask=True,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_background_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                else:
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)

                ids1 = np.arange(self.num_imgs)
                ids2 = ids1 + 1
                ids2[-1] -= 2
                pred_optical_flows_vis, pred_optical_flows = self.get_pred_flows(ids1, ids2, self.depths,
                                                                                 use_max_loc=self.args.use_max_loc,
                                                                                 chunk_size=self.args.chunk_size,
                                                                                 return_original=True
                                                                                 )
                imageio.mimwrite(os.path.join(vis_dir, '{}_flow_{:06d}.mp4'.format(self.seq_name, step)),
                                 pred_optical_flows_vis[:-1],
                                 quality=8, fps=10)

            if self.args.use_error_map and (step % self.args.i_cache == 0) and (step > 0):
                flow_save_dir = os.path.join(self.out_dir, 'flow')
                os.makedirs(flow_save_dir, exist_ok=True)
                flow_errors = []
                for i, (id1, id2) in enumerate(zip(ids1, ids2)):
                    save_path = os.path.join(flow_save_dir, '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                               os.path.basename(self.img_files[id2])))
                    np.save(save_path, pred_optical_flows[i])
                    gt_flow = np.load(os.path.join(self.seq_dir, 'raft_exhaustive',
                                                   '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                      os.path.basename(self.img_files[id2]))
                                                   ))
                    flow_error = np.linalg.norm(gt_flow - pred_optical_flows[i], axis=-1).mean()
                    flow_errors.append(flow_error)

                flow_errors = np.array(flow_errors)
                np.savetxt(os.path.join(self.out_dir, 'flow_error.txt'), flow_errors)

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'deform_mlp': de_parallel(self.deform_mlp).state_dict(),
                   'feature_mlp': de_parallel(self.feature_mlp).state_dict(),
                   'depths': self.depths,
                   'num_imgs': self.num_imgs
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.deform_mlp.load_state_dict(to_load['deform_mlp'])
        self.feature_mlp.load_state_dict(to_load['feature_mlp'])
        self.depths = to_load['depths']
        self.num_imgs = to_load['num_imgs']

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, from scratch...')
            step = 0

        return step
