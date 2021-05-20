import os
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as sn
import torch
import nrrd
import random

class SphericDefectGenerator(object):

  #  This class generates random combinations of synthetic spherical skull defects

    def __init__(self, r1_mean, r1_sdev, r2_mean, r2_sdev, min_sec, max_sec,
                 elastic_alpha=300, elastic_sigma=10, shape=(512, 512, 512)):
      
        self.r1_mean = r1_mean
        self.r1_sdev = r1_sdev
        self.r2_mean = r2_mean
        self.r2_sdev = r2_sdev
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.alpha = elastic_alpha
        self.sigma = elastic_sigma

        self.pregenerated_distmap = np.ones(shape)
        self.pregenerated_distmap[
            shape[0] // 2, shape[1] // 2, shape[2] // 2] = 0
        self.pregenerated_distmap = sn.morphology.distance_transform_edt(
            self.pregenerated_distmap)

    def elastic_transform(self, image, random_state=None):
        assert len(image.shape) == 2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape

        dx = self.alpha * sn.filters.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            self.sigma, mode="constant", cval=0
        )
        dy = self.alpha * sn.filters.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            self.sigma, mode="constant", cval=0
        )

        x, y = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        return sn.interpolation.map_coordinates(
            image, indices, order=1).reshape(shape)

    def generate_defect(self, size):
        distmap_crop = self.pregenerated_distmap

        #  a sphere with random radius for primary defect
        volume_1 = np.zeros(size)
        volume_1[distmap_crop < np.random.normal(self.r1_mean,
                                                 self.r1_sdev)] = 1
        volume_1 = volume_1.astype(np.bool)

        # surface positions on which secondary defects are added
        volume_surface = volume_1 ^ sn.morphology.binary_erosion(volume_1)
        surface_inds = np.where(volume_surface)

        # add random number of secondary defect shapes
        volume_2 = np.ones_like(volume_1)
        for _ in range(np.random.randint(self.min_sec, self.max_sec)):
            ind = np.random.randint(len(surface_inds[0] - 1))
            volume_2[
                surface_inds[0][ind], surface_inds[1][ind], surface_inds[2][
                    ind]] = 0
        volume_2 = sn.morphology.distance_transform_edt(
            volume_2) < np.random.normal(self.r2_mean, self.r2_sdev)

        # final defect shape 
        volume = volume_1 | volume_2
        volume = sn.morphology.binary_opening(volume, iterations=5)

        # random elastic deformation in two planes
        state = np.random.randint(512)
        for i in range(volume.shape[0]):
            if np.amax(volume[i, :, :]) > 0:
                volume[i, :, :] = self.elastic_transform(
                    volume[i, :, :].astype(np.float),
                    random_state=np.random.RandomState(state)
                )
        state = np.random.randint(512)
        for i in range(volume.shape[1]):
            if np.amax(volume[:, i, :]) > 0:
                volume[:, i, :] = self.elastic_transform(
                    volume[:, i, :].astype(np.float),
                    random_state=np.random.RandomState(state)
                )

        return volume > 0.5


def gen_defect_wrapper(image, r1_mean=70, r1_sdev=20, r2_mean=40, r2_sdev=10,
                       min_secondary=1, max_secondary=8, offset=90):
    """
    :param image: nrrd image.
    :param r1_mean: Mean radius of primary sphere.
    :param r1_sdev: STD of primary sphere radius.
    :param r2_mean: Mean radius of secondary spheres.
    :param r2_sdev: STD of secondary spheres radius.
    :param min_secondary: Minimum number of secondary spheres.
    :param max_secondary: Maximum number of secondary spheres.
    :param offset: z offset.
    """
    defect_generator = SphericDefectGenerator(r1_mean, r1_sdev, r2_mean, r2_sdev,
                                       min_secondary, max_secondary,
                                       shape=image.shape)

    x_coords, y_coords, z_coords = np.where(image[:, :, offset:] > 0)
    z_coords += offset

    defect = defect_generator.generate_defect(size=image.shape)

    coord_ind = np.random.randint(len(x_coords))
    point = [x_coords[coord_ind] + np.random.randint(-32, 32),
             y_coords[coord_ind] + np.random.randint(-32, 32),
             z_coords[coord_ind] + np.random.randint(-32, 32)]
    defect_shifted = sn.interpolation.shift(
        defect, (point[0] - image.shape[0] // 2,
                 point[1] - image.shape[1] // 2,
                 point[2] - image.shape[2] // 2),
        order=0)

    data_defective_skull = image.clone().detach() if type(
        image) == torch.Tensor else image.copy()
    data_defective_skull[defect_shifted] = 0

    data_implant = image.clone().detach() if type(
        image) == torch.Tensor else image.copy()
    data_implant[~defect_shifted] = 0

    return data_defective_skull, data_implant

def generate_hole_implants_for_cubic(data,cube_dim):
	x_=data.shape[0]
	y_=data.shape[1]
	z_=data.shape[2]
	full_masking=np.ones(shape=(x_,y_,z_))
	x=random.randint(int(cube_dim/2),x_-int(cube_dim/2))
	y=random.randint(int(cube_dim/2),y_-int(cube_dim/2))
	z=int(z_*(3/4))
	cube_masking=np.zeros(shape=(cube_dim,cube_dim,z_-z))
	full_masking[x-int(cube_dim/2):x+int(cube_dim/2),y-int(cube_dim/2):y+int(cube_dim/2),z:z_]=cube_masking
	return full_masking


def add_cub_defect(img_path, size,ext='.nrrd', ext1='.nii.gz'):

		temp,header=nrrd.read(img_path)

		full_masking=generate_hole_implants_for_cubic(temp,size)
		
		c_masking_1=(full_masking==1)
		c_masking_1=c_masking_1+1-1

		defected_image=c_masking_1*temp

		c_masking=(full_masking==0)
		c_masking=c_masking+1-1
		implants=c_masking*temp

		_, file = os.path.split(img_path)

		def_sk_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'defective_skull', 'cubic',
                               file.replace(ext, '_d' + ext)) # saving defective skull
		implant_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'implant', 'cubic',
                                file.replace(ext, '_i' + ext)) # saving implant 
		def_sk_path_nii = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'defective_skull', 'cubic',
                               file.replace(ext, '_d' + ext1)) 
		implant_path_nii = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'implant', 'cubic',
                                file.replace(ext, '_i' + ext1)) 
		os.makedirs(os.path.join(os.path.dirname(os.path.dirname(img_path)), 'defective_skull',  'cubic'), exist_ok=True)
		os.makedirs(os.path.join(os.path.dirname(os.path.dirname(img_path)), 'implant',  'cubic'), exist_ok=True)
  
		nrrd.write(def_sk_path,defected_image[:,:,0:temp.shape[2]].astype('float64'))
		nrrd.write(implant_path,implants[:,:,0:temp.shape[2]].astype('float64'))

		defect = sitk.ReadImage(def_sk_path)
		sitk.WriteImage(defect, def_sk_path_nii)
		impl = sitk.ReadImage(implant_path)
		sitk.WriteImage(impl, implant_path_nii)


def add_sph_defect(img_path, r1_mean=70, r1_sdev=20, r2_mean=40, r2_sdev=10,
               min_secondary=0, max_secondary=8, num_defects=2, offset=80,
               ext='.nrrd', ext1='.nii.gz'):
    """
    :param img_path: input image path.
    :param r1_mean: Mean radius of primary sphere.
    :param r1_sdev: STD of primary sphere radius.
    :param r2_mean: Mean radius of secondary spheres.
    :param r2_sdev: STD of secondary spheres radius.
    :param min_secondary: Minimum number of secondary spheres.
    :param max_secondary: Maximum number of secondary spheres.
    :param num_defects: How many defects to generate for each case.
    :param ext: Image extension.
    :param offset: z offset.
    """

    sitk_img = sitk.ReadImage(img_path)
    data = sitk.GetArrayFromImage(sitk_img).astype(np.uint8)

    _, file = os.path.split(img_path)

    def_sk_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'defective_skull', 'spherical',
                               file.replace(ext, '_d' + ext)) # saving defective skull
    implant_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'implant', 'spherical',
                                file.replace(ext, '_i' + ext)) # saving implant 
    def_sk_path_nii = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'defective_skull', 'spherical',
                               file.replace(ext, '_d' + ext1)) 
    implant_path_nii = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'implant', 'spherical',
                                file.replace(ext, '_i' + ext1)) 
    
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(img_path)), 'defective_skull',  'spherical'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(img_path)), 'implant',  'spherical'), exist_ok=True)
    print(f'Saving defective_skull and implant for {file}')
    for i in range(num_defects):
        skull, implant = gen_defect_wrapper(data, r1_mean, r1_sdev, r2_mean,
                                            r2_sdev, min_secondary,
                                            max_secondary, offset)

        skull = sitk.GetImageFromArray(skull)
        implant = sitk.GetImageFromArray(implant)
        skull.CopyInformation(sitk_img)
        implant.CopyInformation(sitk_img)

        sitk.WriteImage(skull, def_sk_path.replace('_d', f'_d{i}'))
        sitk.WriteImage(implant, implant_path.replace('_i', f'_i{i}'))
        sitk.WriteImage(skull, def_sk_path_nii.replace('_d', f'_d{i}'))
        sitk.WriteImage(implant, implant_path_nii.replace('_i', f'_i{i}'))

def get_num(a, b, x):
    if not a % x:
        return random.choice(range(a, b, x))
    else:
        return random.choice(range(a + x - (a%x), b, x))

def add_defect2folder(folder, num_defects=3, ext='.nrrd', ext1 = '.nii.gz'):
    for file in os.listdir(folder):
        if not file.endswith(ext):
            continue
        path = os.path.join(folder, file)
        size = get_num(1, 128, 2)
        add_cub_defect(path, size, ext = ext, ext1 = ext1)
        add_sph_defect(path, num_defects = num_defects, ext = ext, ext1 = ext1)



if __name__ == '__main__':
    folder = './trainset2021/complete_skull'
    add_defect2folder(folder, num_defects=1)