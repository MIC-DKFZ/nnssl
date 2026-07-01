from batchgenerators.transforms.abstract_transforms import AbstractTransform
from scipy.special import comb
import numpy as np
import torch

from nnssl.training.nnsslTrainer.gvsl.gvsl_specific_modules import DeformableTransformer, AffineTransformer


class GVSLTransform(AbstractTransform):
    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.appearance_transforms = AppearanceTransforms()

    def __call__(self, **data_dict):
        imgs = data_dict.get(self.data_key)
        if imgs is None:
            raise ValueError(f"No data found for key {self.data_key}")

        # The batch size in GVSL can be misleading, since two batches are sampled for each train iteration,
        # resulting in 2*batch_size images. Hence, the dataloader in GVSLTrainer samples 2*batch_size images,
        # which have to be split into 2 equal sized batches with the actual batch_size first.
        batch_size = imgs.shape[0] // 2
        imgsA, imgsB = imgs[:batch_size], imgs[batch_size:]
        assert len(imgsA) == len(imgsB)

        imgsA_app = self.appearance_transforms.rand_aug(imgsA.copy())

        data_dict.update(
            {
                "imgsA": imgsA,
                "imgsA_app": imgsA_app,
                "imgsB": imgsB
            }
        )
        return data_dict

class SpatialTransforms(object):
    # orig
    # def __init__(self, do_rotation=True, angle_x=(-np.pi / 9, np.pi / 9), angle_y=(-np.pi / 9, np.pi / 9),
    #              angle_z=(-np.pi / 9, np.pi / 9), do_scale=True, scale_x=(0.75, 1.25), scale_y=(0.75, 1.25),
    #              scale_z=(0.75, 1.25), do_translate=True, trans_x=(-0.1, 0.1), trans_y=(-0.1, 0.1), trans_z=(-0.1, 0.1),
    #              do_shear=True, shear_xy=(-np.pi / 18, np.pi / 18), shear_xz=(-np.pi / 18, np.pi / 18),
    #              shear_yx=(-np.pi / 18, np.pi / 18), shear_yz=(-np.pi / 18, np.pi / 18),
    #              shear_zx=(-np.pi / 18, np.pi / 18), shear_zy=(-np.pi / 18, np.pi / 18),
    #              do_elastic_deform=True, alpha=(0., 512.), sigma=(10., 13.)):

    def __init__(self, do_rotation=True, angle_x=(-np.pi / 9, np.pi / 9), angle_y=(-np.pi / 9, np.pi / 9),
                 angle_z=(-np.pi / 9, np.pi / 9), do_scale=True, scale_x=(0.75, 1.25), scale_y=(0.75, 1.25),
                 scale_z=(0.75, 1.25), do_translate=True, trans_x=(-0.1, 0.1), trans_y=(-0.1, 0.1), trans_z=(-0.1, 0.1),
                 do_shear=True, shear_xy=(-np.pi / 18, np.pi / 18), shear_xz=(-np.pi / 18, np.pi / 18),
                 shear_yx=(-np.pi / 18, np.pi / 18), shear_yz=(-np.pi / 18, np.pi / 18),
                 shear_zx=(-np.pi / 18, np.pi / 18), shear_zy=(-np.pi / 18, np.pi / 18),
                 do_elastic_deform=True, alpha=(0., 512.), sigma=(10., 13.)):

        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z

        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_translate = do_translate
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.trans_z = trans_z
        self.do_shear = do_shear
        self.shear_xy = shear_xy
        self.shear_xz = shear_xz
        self.shear_yx = shear_yx
        self.shear_yz = shear_yz
        self.shear_zx = shear_zx
        self.shear_zy = shear_zy

        self.deformable_transformer = DeformableTransformer()
        self.affine_transformer = AffineTransformer()

    def augment_spatial(self, data, affine_mat=None, deformable_map=None, mode='bilinear'):
        if affine_mat is not None:
            data = self.affine_transformer(data, affine_mat, mode)
        if deformable_map is not None:
            data = self.deformable_transformer(data, deformable_map, mode=mode, padding_mode='zeros')
        return data

    def get_rand_spatial(self, batch_size: int, patch_size: tuple[int] | np.ndarray):
        affine_mat = []
        flow = []
        for i in range(batch_size):
            _affine_mat = torch.eye(3) #.cuda()
            _flow = self.create_zero_centered_coordinate_mesh(patch_size) #.cuda()
            if self.do_rotation:
                a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
                a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
                a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
                _affine_mat = self.rotate_mat(_affine_mat, a_x, a_y, a_z)
            if self.do_scale:
                sc_x = np.random.uniform(self.scale_x[0], self.scale_x[1])
                sc_y = np.random.uniform(self.scale_y[0], self.scale_y[1])
                sc_z = np.random.uniform(self.scale_z[0], self.scale_z[1])
                _affine_mat = self.scale_mat(_affine_mat, sc_x, sc_y, sc_z)
            if self.do_shear:
                s_xy = np.random.uniform(self.shear_xy[0], self.shear_xy[1])
                s_xz = np.random.uniform(self.shear_xz[0], self.shear_xz[1])
                s_yx = np.random.uniform(self.shear_yx[0], self.shear_yx[1])
                s_yz = np.random.uniform(self.shear_yz[0], self.shear_yz[1])
                s_zx = np.random.uniform(self.shear_zx[0], self.shear_zx[1])
                s_zy = np.random.uniform(self.shear_zy[0], self.shear_zy[1])
                _affine_mat = self.shear_mat(_affine_mat, s_xy, s_xz, s_yx, s_yz, s_zx, s_zy)
            if self.do_translate:
                t_x = np.random.uniform(self.trans_x[0], self.trans_x[1]) * patch_size[0]
                t_y = np.random.uniform(self.trans_y[0], self.trans_y[1]) * patch_size[1]
                t_z = np.random.uniform(self.trans_z[0], self.trans_z[1]) * patch_size[2]
                _affine_mat = self.translate_mat(_affine_mat, t_x, t_y, t_z)
            else:
                _affine_mat = self.translate_mat(_affine_mat, 0, 0, 0)
            if self.do_elastic_deform:
                a = np.random.uniform(self.alpha[0], self.alpha[1])
                s = np.random.uniform(self.sigma[0], self.sigma[1])
                _flow = self.deform_coords(_flow, a, s)

            center = torch.tensor([patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]) #.cuda()

            vectors = [torch.arange(0, s) for s in patch_size]
            grids = torch.meshgrid(vectors, indexing="ij")
            grid = torch.stack(grids).float() #.cuda()

            _affine_mat = _affine_mat.unsqueeze(0)
            _flow += center.reshape(1, -1, 1, 1, 1) - grid.unsqueeze(0)

            affine_mat.append(_affine_mat)
            flow.append(_flow)

        affine_mat = torch.concat(affine_mat, 0)
        flow = torch.concat(flow, 0)

        return affine_mat, flow

    def create_zero_centered_coordinate_mesh(self, shape):
        tmp = [torch.arange(i) for i in shape]
        coords = torch.stack(torch.meshgrid(*tmp, indexing='ij')).float()
        offset = (torch.tensor(shape) - 1.) / 2.
        for d in range(len(shape)):
            coords[d] -= offset[d]
        return coords.unsqueeze(0)

    def rotate_mat(self, mat, angle_x, angle_y, angle_z):
        rot_mat_x = torch.tensor([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]], dtype=torch.float32) #.cuda()
        rot_mat_y = torch.tensor([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]], dtype=torch.float32) #.cuda()
        rot_mat_z = torch.tensor([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]], dtype=torch.float32) #.cuda()
        mat = torch.matmul(rot_mat_z, torch.matmul(rot_mat_y, torch.matmul(rot_mat_x, mat)))
        return mat

    def deform_coords(self, coords, alpha, sigma):
        offsets = torch.rand(coords.shape)*2-1 #.cuda()*2 -1
        ker1d = self._gaussian_kernel1d(sigma).reshape(1, 1, -1)
        ker1d1 = ker1d[:, :, :, None, None]
        ker1d2 = ker1d[:, :, None, :, None]
        ker1d3 = ker1d[:, :, None, None, :]

        for i in range(3):
            offsets[:, i:i+1] = torch.conv3d(input=offsets[:, i:i+1], weight=ker1d1, padding=[ker1d.shape[-1]//2, 0, 0])
            offsets[:, i:i+1] = torch.conv3d(input=offsets[:, i:i+1], weight=ker1d2, padding=[0, ker1d.shape[-1]//2, 0])
            offsets[:, i:i+1] = torch.conv3d(input=offsets[:, i:i+1], weight=ker1d3, padding=[0, 0, ker1d.shape[-1]//2])
        offsets = offsets * alpha
        indices = offsets + coords
        return indices

    def _gaussian_kernel1d(self, sigma):
        sd = float(sigma)
        radius = int(4 * sd + 0.5)
        sigma2 = sigma ** 2
        x = torch.arange(-radius, radius + 1)
        phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()

        return phi_x


    def scale_mat(self, mat, scale_x, scale_y, scale_z):
        scale_mat = torch.tensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]]) #.cuda()
        mat = torch.matmul(scale_mat, mat)
        return mat

    def shear_mat(self, mat, shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy):
        shear_mat = torch.tensor([[1, np.tan(shear_xy), np.tan(shear_xz)], [np.tan(shear_yx), 1, np.tan(shear_yz)],
                     [np.tan(shear_zx), np.tan(shear_zy), 1]], dtype=torch.float32) #.cuda()
        mat = torch.matmul(shear_mat, mat)
        return mat

    def translate_mat(self, mat, trans_x, trans_y, trans_z):
        trans = torch.tensor([trans_x, trans_y, trans_z]) #.cuda()
        trans = trans[:, np.newaxis]
        mat = torch.cat([mat, trans], dim=-1)
        return mat


class AppearanceTransforms(object):
    def __init__(self, local_rate=0.8, nonlinear_rate=0.9, paint_rate=0.9, inpaint_rate=0.2, is_local=True, is_nonlinear=True, is_in_painting=True):
        self.is_local = is_local
        self.is_nonlinear = is_nonlinear
        self.is_in_painting = is_in_painting
        self.local_rate = local_rate
        self.nonlinear_rate = nonlinear_rate

        self.paint_rate = paint_rate
        self.inpaint_rate = inpaint_rate


    def rand_aug(self, imgs):
        _imgs = imgs.copy()
        for i in range(len(imgs)):
            if self.is_local:
                _imgs[i] = self.local_pixel_shuffling(_imgs[i], prob=self.local_rate)
            if self.is_nonlinear:
                _imgs[i] = self.nonlinear_transformation(_imgs[i], self.nonlinear_rate)
            if self.is_in_painting:
                _imgs[i] = self.image_in_painting(_imgs[i])
        _imgs = _imgs.astype(np.float32)
        return _imgs


    def bernstein_poly(self, i, n, t):
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def nonlinear_transformation(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        points = [[0, 0], [np.random.random(), np.random.random()], [np.random.random(), np.random.random()], [1, 1]]

        xvals, yvals = self.bezier_curve(points, nTimes=100000)

        xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x

    def local_pixel_shuffling(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        image_temp = x.copy()
        orig_image = x.copy()
        _, img_rows, img_cols, img_deps = x.shape
        num_block = 5000
        block_noise_size_x = int(img_rows // 20)
        block_noise_size_y = int(img_cols // 20)
        block_noise_size_z = int(img_deps // 20)
        noise_x = np.random.randint(low=img_rows - block_noise_size_x, size=num_block)
        noise_y = np.random.randint(low=img_cols - block_noise_size_y, size=num_block)
        noise_z = np.random.randint(low=img_deps - block_noise_size_z, size=num_block)
        window=[orig_image[:, noise_x[i]:noise_x[i] + block_noise_size_x, noise_y[i]:noise_y[i] + block_noise_size_y,
                     noise_z[i]:noise_z[i] + block_noise_size_z,] for i in range(num_block)]
        window = np.concatenate(window, axis=0)
        window = window.reshape(num_block, -1)
        np.random.shuffle(window.T)
        window = window.reshape((num_block, block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        for i in range(num_block):
            image_temp[0, noise_x[i]:noise_x[i] + block_noise_size_x,
            noise_y[i]:noise_y[i] + block_noise_size_y,
            noise_z[i]:noise_z[i] + block_noise_size_z] = window[i]
        local_shuffling_x = image_temp

        return local_shuffling_x

    def image_in_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        cnt = 30
        while cnt > 0 and np.random.random() < 0.95:
            block_noise_size_x = np.random.randint(img_rows // 10, img_rows // 5)
            block_noise_size_y = np.random.randint(img_cols // 10, img_cols // 5)
            block_noise_size_z = np.random.randint(img_deps // 10, img_deps // 5)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x