import argparse
import math
import os
import ants
import tempfile
import textwrap
from os import path as osp
import SimpleITK as sitk
import numpy as np
import pandas as pd
import glob
import copy
import random

def veri_folder(path=None):
    if not path:
        return None
    else:
        os.makedirs(path, exist_ok=True)
        return path

def get_largest_cc(image):

  #  Retains only the largest connected component of a binary image

    image = sitk.Cast(image, sitk.sitkUInt32)

    connectedComponentFilter = sitk.ConnectedComponentImageFilter()
    objects = connectedComponentFilter.Execute(image)

    # If there is more than one connected component
    if connectedComponentFilter.GetObjectCount() > 1:
        objectsData = sitk.GetArrayFromImage(objects)

        # Detect the largest connected component
        maxLabel = 1
        maxLabelCount = 0
        for i in range(1, connectedComponentFilter.GetObjectCount() + 1):
            componentData = objectsData[objectsData == i]

            if len(componentData.flatten()) > maxLabelCount:
                maxLabel = i
                maxLabelCount = len(componentData.flatten())

        # Remove all the values, exept the ones for the largest connected component

        dataAux = np.zeros(objectsData.shape, dtype=np.uint8)

        # Fuse the labels

        dataAux[objectsData == maxLabel] = 1

        # Save edited image
        output = sitk.GetImageFromArray(dataAux)
        output.SetSpacing(image.GetSpacing())
        output.SetOrigin(image.GetOrigin())
        output.SetDirection(image.GetDirection())
    else:
        output = image

    return output

def register_ant(moving_img_path, fixed_image, out_image=None,
                    mat_path=None, transformation='QuickRigid',
                    overwrite=False):
    atlas_name = osp.split(fixed_image)[1].split(".")[0]
    mv_img_p, mv_img_n = osp.split(moving_img_path)

    if osp.exists(out_image) and not overwrite and osp.getsize(out_image) > 0:
        print("\n  The output file already exists (skipping).")
        return
    else:
        ct = ants.image_read(moving_img_path)
        fixed = ants.image_read(fixed_image)
        my_tx = ants.registration(fixed=fixed, moving=ct,
                                  type_of_transform='QuickRigid')
        transf_ct = my_tx['warpedmovout']
        reg_transform = my_tx['fwdtransforms']
        ants.write_transform(ants.read_transform(reg_transform[0]), mat_path)
        ants.image_write(transf_ct, out_image)


def register_ants_sitk(moving_image, fixed_image=None, mat_save_path=None, save_path ="",
                       transformation="QuickRigid", interp="nearestNeighbor",
                       mat_to_apply=None, reference=None):
  
    mv_img = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    mv_img_path = mv_img.name

    sitk.WriteImage(moving_image, mv_img_path)
    
    if fixed_image and type(fixed_image) is not str:
        fx_img = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        fx_img_path = fx_img.name
        sitk.WriteImage(fixed_image, fx_img_path)
    elif fixed_image:
        fx_img_path = fixed_image

    register_ant(mv_img_path, fx_img_path, save_path, mat_save_path,
                         transformation)

    ret_img = sitk.ReadImage(save_path)
    return ret_img


class Preprocessor:
    def __init__(self, sitk_img=None, is_binary=False, save_path="",
                 image_path=None):
        self.image = sitk_img if sitk_img else None

        self.image_path = ""
        self.is_binary = is_binary

        self.direction = None
        self.origin = None

        self.load_image(image_path)

        if os.path.splitext(save_path)[1] == "":
            self.save_path = veri_folder(save_path)
        else: 
            veri_folder(os.path.split(save_path)[0])
            self.save_path = save_path

    def load_image(self, image_path=None):

        print(f"Image: {image_path}\n")
        self.image = sitk.ReadImage(image_path) if image_path else None

        self.image_path = image_path

        if self.image:
            self.direction = self.image.GetDirection()
            self.origin = self.image.GetOrigin()
            self.spacing = self.image.GetSpacing()

    def set_filename(self, file_name=None):

        if file_name:
            self.image_path = file_name
            return file_name
        else:
            return self.image_path

    def set_save_path(self, save_path=None):

        if save_path:
            self.save_path = save_path

    def save_file(self):

        if self.save_path is None:
            return None

        save_path_image = self.save_path
        sitk.WriteImage(self.image, save_path_image)  
        print("\n   Preprocessed image saved in {}.".format(save_path_image))

        return save_path_image


    def register_antspy(self, fixed_im_path, save_transform=False,
                        transformation="QuickRigid", apply=True,
                        img_interp="nearestNeighbor"):
        if not apply:
            return None

        print("ANTsPy Registering")
        print(f"Fixed image: {fixed_im_path}")


        if save_transform is not None:
            mat_path = os.path.join(
                (self.save_path if os.path.isdir(self.save_path) else
                 os.path.split(self.save_path)[0]),
                os.path.split(self.image_path)[1].replace('nii.gz', '') + "_reg.mat"
            )
            print(f"Transformation will be saved in {mat_path}")


        if self.image:
            print("Registering image")

            self.image = register_ants_sitk(self.image,
                                            fixed_im_path,
                                            mat_path,
                                            self.save_path,
                                            img_interp,
                                            transformation)

    def keep_largest_cc(self, apply=True):
        if apply:
            self.image = get_largest_cc(self.image)


def prep_image(image_path=None, output_ff=None, 
                       clip_intensity_values=None, target_spacing=None,
                       fixed_size_pad=None, threshold=False, largest_cc=False,
                       register=True, transformation='QuickRigid',
                       random_blank_patch=False, img_is_binary=False,
                       atlas_path=None, img_interp='nearestNeighbor'):

    pp = Preprocessor(save_path=output_ff, image_path=image_path, is_binary=img_is_binary)
    pp.register_antspy(atlas_path, True, transformation, register, img_interp)
    pp.keep_largest_cc(largest_cc)
    if not pp.save_path:
        return pp.image 
    else:
        pp.save_file()


def prep_img_autoimpl(input_ff,  n, zone, overwrite=False):
    skulls = glob.glob(f"{input_ff}/*.nii.gz")
    cases1 = sorted([os.path.basename(s).split(".")[0] for s in random.sample(skulls, n)])
    cases2 = copy.copy(cases1)
    for i, case1 in enumerate(cases1):
        for case2 in cases2[(i+1):]:
            for name in (f"defective_skull/{zone}", f"implant/{zone}", "complete_skull"):  
                  fixed = f'./trainset2021/{name}/{case1}.nii.gz'
                  moving = f'./trainset2021/{name}/{case2}.nii.gz'
                  output_ff = f'./trainset2021/{name}/{case2}_to_{case1}.nii.gz'
                  prep_image(image_path = moving, output_ff = output_ff, 
                           clip_intensity_values = None, target_spacing = None,
                           fixed_size_pad = None, threshold = False, largest_cc = True, register = True,
                           transformation = 'QuickRigid', random_blank_patch = False, img_is_binary=False, atlas_path = fixed, img_interp = 'nearestNeighbor')


if __name__ == '__main__':
    prep_img_autoimpl('./trainset2021/complete_skull', 10, 'bilateral', overwrite=False)
    prep_img_autoimpl('./trainset2021/complete_skull', 10, 'frontoorbital', overwrite=False)
    prep_img_autoimpl('./trainset2021/complete_skull', 10, 'parietotemporal', overwrite=False)
    prep_img_autoimpl('./trainset2021/complete_skull', 10, 'random_1', overwrite=False)
    prep_img_autoimpl('./trainset2021/complete_skull', 10, 'random_2', overwrite=False)
