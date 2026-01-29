from .quimb.vm_ring_model import *
from .quimb.tn_utils import *
from .tensorBase import *

class TensorVM_Ring(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM_Ring, self).__init__(aabb, gridSize, device, **kargs)
        self.num_upsamples_perfomed = 0
        print("====> init TensoRing - fused: ", self.fused)
    

    def init_svd_volume(self, res, device):
        self.init_res = res
        # max_rank_appearance and max_rank_density and max_rank are the possibilities
        if self.fused:
            payload = self.app_dim + 1
            self.vox_fused = vmring3dQuimb(res, max_rank_tt = self.max_rank, use_TTNF_sampling= self.use_TTNF_sampling,
                                        payload_dim = payload,
                                        compression_alg = self.compression_alg, canonization = self.canonization)
        else:
            max_rank_appearance = self.max_rank_appearance if self.max_rank_appearance > 0 else self.max_rank
            max_rank_density = self.max_rank_density if self.max_rank_density > 0 else self.max_rank

            self.vox_rgb_plane = vmring3dQuimb(res, max_rank_tt = max_rank_appearance, use_TTNF_sampling= self.use_TTNF_sampling,
                                    payload_dim = self.app_n_comp[0],
                                    compression_alg = self.compression_alg, canonization = self.canonization,isplane=True)  # color
            self.vox_rgb_plane1 = vmring3dQuimb(res, max_rank_tt = max_rank_appearance, use_TTNF_sampling= self.use_TTNF_sampling,
                                    payload_dim = self.app_n_comp[1],
                                    compression_alg = self.compression_alg, canonization = self.canonization,isplane=True)  # color
            self.vox_rgb_plane2 = vmring3dQuimb(res, max_rank_tt = max_rank_appearance, use_TTNF_sampling= self.use_TTNF_sampling,
                                    payload_dim = self.app_n_comp[2],
                                    compression_alg = self.compression_alg, canonization = self.canonization,isplane=True)  # color
            self.vox_rgb_line = vmring3dQuimb(res, max_rank_tt = max_rank_appearance, use_TTNF_sampling= self.use_TTNF_sampling,
                                    payload_dim = self.app_n_comp[0],
                                    compression_alg = self.compression_alg, canonization = self.canonization)  # color
            self.vox_rgb_line1 = vmring3dQuimb(res, max_rank_tt = max_rank_appearance, use_TTNF_sampling= self.use_TTNF_sampling,
                                    payload_dim = self.app_n_comp[1],
                                    compression_alg = self.compression_alg, canonization = self.canonization)  # color
            self.vox_rgb_line2 = vmring3dQuimb(res, max_rank_tt = max_rank_appearance, use_TTNF_sampling= self.use_TTNF_sampling,
                                    payload_dim = self.app_n_comp[2],
                                    compression_alg = self.compression_alg, canonization = self.canonization)  # color
            
            self.vox_sigma_plane = vmring3dQuimb(res, max_rank_tt = max_rank_density, use_TTNF_sampling=self.use_TTNF_sampling,
                                    payload_dim = self.density_n_comp[0],
                                    compression_alg = self.compression_alg, canonization = self.canonization,isplane=True)
            self.vox_sigma_plane1 = vmring3dQuimb(res, max_rank_tt = max_rank_density, use_TTNF_sampling=self.use_TTNF_sampling,
                                    payload_dim = self.density_n_comp[1],
                                    compression_alg = self.compression_alg, canonization = self.canonization,isplane=True)
            self.vox_sigma_plane2 = vmring3dQuimb(res, max_rank_tt = max_rank_density, use_TTNF_sampling=self.use_TTNF_sampling,
                                    payload_dim = self.density_n_comp[2],
                                    compression_alg = self.compression_alg, canonization = self.canonization,isplane=True)
            self.vox_sigma_line = vmring3dQuimb(res, max_rank_tt = max_rank_density, use_TTNF_sampling=self.use_TTNF_sampling,
                                    payload_dim = self.density_n_comp[0],
                                    compression_alg = self.compression_alg, canonization = self.canonization)
            self.vox_sigma_line1 = vmring3dQuimb(res, max_rank_tt = max_rank_density, use_TTNF_sampling=self.use_TTNF_sampling,
                                    payload_dim = self.density_n_comp[1],
                                    compression_alg = self.compression_alg, canonization = self.canonization)
            self.vox_sigma_line2 = vmring3dQuimb(res, max_rank_tt = max_rank_density, use_TTNF_sampling=self.use_TTNF_sampling,
                                    payload_dim = self.density_n_comp[2],
                                    compression_alg = self.compression_alg, canonization = self.canonization)
            self.basis_mat = torch.nn.Linear(self.app_n_comp[0]+self.app_n_comp[1]+self.app_n_comp[2], self.app_dim, bias=False, device=device).to(device)  
        self.take_voxel_representations_on_device(device)
        

    def take_voxel_representations_on_device(self, device):
        """
        Transfers the voxel representations to the specified device.

        Parameters:
        device (str): The device to transfer the voxel representations to.
        """
        if self.fused:
            self.vox_fused = self.vox_fused.to(device) 
        else:
            self.vox_rgb_plane = self.vox_rgb_plane.to(device)
            self.vox_rgb_plane1 = self.vox_rgb_plane1.to(device)
            self.vox_rgb_plane2 = self.vox_rgb_plane2.to(device)
            self.vox_rgb_line = self.vox_rgb_line.to(device)
            self.vox_rgb_line1 = self.vox_rgb_line1.to(device)
            self.vox_rgb_line2 = self.vox_rgb_line2.to(device)
            self.vox_sigma_plane = self.vox_sigma_plane.to(device)
            self.vox_sigma_plane1 = self.vox_sigma_plane1.to(device)
            self.vox_sigma_plane2 = self.vox_sigma_plane2.to(device)
            self.vox_sigma_line = self.vox_sigma_line.to(device)
            self.vox_sigma_line1 = self.vox_sigma_line1.to(device)
            self.vox_sigma_line2 = self.vox_sigma_line2.to(device)
            

    def compute_features(self, xyz_sampled):
        """
        Computes features for the given sampled coordinates.

        Parameters:
        xyz_sampled (array): Sampled coordinates with shape (N, 3).

        Returns:
        tuple: The computed RGB and sigma values with shapes (N, 3) and (N,).
        """
        if self.fused:
            res =  self.vox_fused(xyz_sampled)
            # devide into rgb and sigma
            rgb = res[:, :-1]
            sigma = res[:, -1:]
            return rgb, sigma.squeeze()
        else:
            rgb = self.vox_rgb(xyz_sampled)
            sigma = self.vox_sigma(xyz_sampled)
            return rgb, sigma.squeeze()

    
    def compute_densityfeature(self, xyz_sampled):
        """
        Computes the density feature for the given sampled coordinates.

        Parameters:
        xyz_sampled (array): Sampled coordinates with shape (N, 3).

        Returns:
        array: The computed density values with shape (N,).
        """
        # print min max of xyz_sampled
        if self.fused:
            _, sigma = self.compute_features(xyz_sampled)
            return sigma.squeeze()
        #sigma_feature = torch.sum(line_coef_point, dim=0)
        else:
            plane_coef = [
                self.vox_sigma_plane().unsqueeze(0),
                self.vox_sigma_plane1().unsqueeze(0),
                self.vox_sigma_plane2().unsqueeze(0)
            ]

            line_coef = [
                self.vox_sigma_line().unsqueeze(0).unsqueeze(-1),
                self.vox_sigma_line1().unsqueeze(0).unsqueeze(-1),
                self.vox_sigma_line2().unsqueeze(0).unsqueeze(-1)
            ]
            coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)#shape3,N,1,2
            coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)#shape3,N,1,2
            sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
            for idx_plane in range(len(plane_coef)):
                plane_coef_point = F.grid_sample(plane_coef[idx_plane], coordinate_plane[[idx_plane]],
                                                    align_corners=True).view(-1, *xyz_sampled.shape[:1])
                line_coef_point = F.grid_sample(line_coef[idx_plane], coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
                sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
            return sigma_feature
            
            plane_coef=torch.stack(
                (self.vox_sigma_plane(),self.vox_sigma_plane1(),self.vox_sigma_plane2()),
                dim=0)
            line_coef=torch.stack(
                (self.vox_sigma_line(),self.vox_sigma_line1(),self.vox_sigma_line2()),
                dim=0).unsqueeze(-1)
            coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)#shape3,N,1,2
            coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)#shape3,N,1,2
        
            plane_feats = F.grid_sample(plane_coef, coordinate_plane, align_corners=True).view(3 * self.density_n_comp, -1)#plane_coef3,48,res,res
            line_feats = F.grid_sample(line_coef, coordinate_line, align_corners=True).view(3 * self.density_n_comp, -1)
        
        
            sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
            return sigma_feature
        
    
    def compute_appfeature(self, xyz_sampled):
        """
        Computes the appearance feature for the given sampled coordinates.

        Parameters:
        xyz_sampled (array): Sampled coordinates with shape (N, 3).

        Returns:
        array: The computed RGB values with shape (N, 3).
        """
        if self.fused:
            rgb, _ = self.compute_features(xyz_sampled)
            return rgb
        else:
            plane_coef = [
                self.vox_rgb_plane().unsqueeze(0),
                self.vox_rgb_plane1().unsqueeze(0),
                self.vox_rgb_plane2().unsqueeze(0)
            ]
            line_coef = [
                self.vox_rgb_line().unsqueeze(0).unsqueeze(-1),
                self.vox_rgb_line1().unsqueeze(0).unsqueeze(-1),
                self.vox_rgb_line2().unsqueeze(0).unsqueeze(-1)
            ]
            coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)#shape3,N,1,2
            coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)#shape3,N,1,2
            plane_coef_point,line_coef_point = [],[]
            for idx_plane in range(len(plane_coef)):
                plane_coef_point.append(F.grid_sample(plane_coef[idx_plane], coordinate_plane[[idx_plane]],
                                                    align_corners=True).view(-1, *xyz_sampled.shape[:1]))
                line_coef_point.append(F.grid_sample(line_coef[idx_plane], coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


            return self.basis_mat((plane_coef_point * line_coef_point).T)
    
    def density_L1(self):
        """
        Calculates the L1 norm of the density tensors.

        Returns:
        float: The computed L1 norm.
        """
        total = 0
        tensors = (self.vox_sigma_plane.tn.tensors + self.vox_sigma_plane1.tn.tensors + self.vox_sigma_plane2.tn.tensors+self.vox_sigma_line.tn.tensors + self.vox_sigma_line1.tn.tensors + self.vox_sigma_line2.tn.tensors)
        for i in range(len(tensors)):
            total += torch.mean(torch.abs(tensors[i].data))
        
        return total
      
    
    def density_L2(self):
        """
        Calculates the L1 norm of the density tensors.

        Returns:
        float: The computed L1 norm.
        """
        total = 0
        tensors = self.vox_sigma.tn.tensors
        for i in range(len(tensors)):
            
            tensor_vectorized = tensors[i].data.contiguous().view(-1)
            score = torch.matmul(tensor_vectorized, tensor_vectorized)
            total += torch.mean(torch.sqrt(score))
        return total
    def TV_loss(self, density_only = False):
        """
        Computes the total variation loss for the volume grid of either the RGB or the density tensors.
        """
        if density_only:
            return self.vox_sigma.compute_total_variation_loss()
        else:
            return self.vox_rgb.compute_total_variation_loss() 
    
    def get_max_ranks(self):
        """
        Retrieves the maximum ranks of the volume grid.

        Returns:
        tuple: The maximum ranks of the RGB and the density tensors.
        """
        if self.fused:
            return self.vox_fused.tn.max_bond(), self.vox_fused.tn.max_bond()
        else:
            return self.vox_rgb.tn.max_bond(), self.vox_sigma.tn.max_bond()
        
    def get_norms(self):
        """
        Retrieves the Frobenius norms of the volume grid.

        Returns:
        tuple: The Frobenius norms of the RGB and the density tensors.
        """
        if self.fused:
            # return self.vox_fused.tn.H @ self.vox_fused.tn 
            return self.vox_fused.tn.norm(), self.vox_fused.tn.norm()
        else:
            # norm_vox_rgb = self.vox_rgb.tn.H @ self.vox_rgb.tn
            # norm_vox_sigma = self.vox_sigma.tn.H @ self.vox_sigma.tn
            # return norm_vox_rgb, norm_vox_sigma
            return self.vox_rgb.tn.norm(), self.vox_sigma.tn.norm()
    
    @torch.no_grad()
    def upsample_volume_ranks(self, max_rank_appearance, max_rank_density):
        """
        Upsamples the volume grid ranks to the target rank.

        Parameters:
        max_rank (int): Target rank.
        """
        if self.fused:
            self.vox_fused.upsample_vmting_ranks(max_rank_appearance)
            self.vox_fused.max_rank_tt = max_rank_appearance
        else:
            self.vox_rgb.upsample_vmring_ranks(max_rank_appearance)
            self.vox_sigma.upsample_vmring_ranks(max_rank_density)
            self.vox_rgb.max_rank_tt = max_rank_appearance
            self.vox_sigma.max_rank_tt = max_rank_density
            
            print(" sigma", self.vox_sigma.tn)
            print(" rgb", self.vox_rgb.tn)

    def upsample_vmring_volume_ranks(self):
        """
        Upsamples the volume grid ranks to the target rank.

        Parameters:
        max_rank (int): Target rank.
        """
        if self.fused:
            self.vox_fused.upsample_vmring_ranks()
        else:
            self.vox_rgb.upsample_mera_ranks()
            self.vox_sigma.upsample_mera_ranks()
            print(" sigma", self.vox_sigma.tn)
            print(" rgb", self.vox_rgb.tn)
                
    
    @torch.no_grad()
    def upsample_volume_grid(self, reso_target):
        """
        Upsamples the volume grid to the target resolution.

        Parameters:
        reso_target (int): Target resolution.
        """
        if self.fused:
            self.vox_fused.upsample(self.num_upsamples_perfomed)
        else:
            self.vox_rgb_plane.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_rgb_plane1.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_rgb_plane2.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_rgb_line.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_rgb_line1.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_rgb_line2.upsample(self.num_upsamples_perfomed,reso_target)

            self.vox_sigma_plane.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_sigma_plane1.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_sigma_plane2.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_sigma_line.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_sigma_line1.upsample(self.num_upsamples_perfomed,reso_target)
            self.vox_sigma_line2.upsample(self.num_upsamples_perfomed,reso_target)
        self.num_upsamples_perfomed += 1


        self.update_stepSize(reso_target)
        print(f'upsamping to {reso_target}')


    def get_optparam_groups(self, lr_init_spatial = 0.003, lr_init_network = 0.001):
        """
        Groups the optimization parameters.

        Parameters:
        lr_init_spatial (float): Initial learning rate for spatial parameters.
        lr_init_network (float): Initial learning rate for network parameters.

        Returns:
        list: List of dictionaries containing parameters and their learning rates.
        """
        out = []
        if self.fused:
            out += [
                {'params': self.vox_fused.parameters(), 'lr': lr_init_spatial}
            ]
        else:
            out += [
                {'params': self.vox_rgb_plane.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_rgb_plane1.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_rgb_plane2.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_rgb_line.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_rgb_line1.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_rgb_line2.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_sigma_plane.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_sigma_plane1.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_sigma_plane2.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_sigma_line.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_sigma_line1.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_sigma_line2.parameters(), 'lr': lr_init_spatial},
                {'params': self.basis_mat.parameters(), 'lr':lr_init_network}
            ]
        if isinstance(self.renderModule, torch.nn.Module):
            out += [
                {'params': self.renderModule.parameters(), 'lr': lr_init_network}
            ]
        return out

    def get_compression_values(self):
        """
        Retrieves various compression values.

        Returns:
        dict: A dictionary containing uncompressed and compressed parameters, sizes, and compression factor.
        """
        if self.fused:
            self.num_uncompressed_params = self.vox_fused.num_uncompressed_params
            self.num_compressed_params = self.vox_fused.num_compressed_params
            self.sz_uncompressed_gb = self.vox_fused.sz_uncompressed_gb
            self.sz_compressed_gb = self.vox_fused.sz_compressed_gb
        else:
            self.num_uncompressed_params = self.vox_rgb.num_uncompressed_params + self.vox_sigma.num_uncompressed_params
            self.num_compressed_params =  self.vox_rgb.num_compressed_params + self.vox_sigma.num_compressed_params
            self.sz_uncompressed_gb = self.vox_rgb.sz_uncompressed_gb + self.vox_sigma.sz_uncompressed_gb
            self.sz_compressed_gb = self.vox_rgb.sz_compressed_gb + self.vox_sigma.sz_compressed_gb
            
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params
        return {
            'num_uncompressed_params': self.num_uncompressed_params,
            'num_compressed_params': self.num_compressed_params,
            'sz_uncompressed_gb': self.sz_uncompressed_gb,
            'sz_compressed_gb': self.sz_compressed_gb,
            'compression_factor': self.compression_factor
        }