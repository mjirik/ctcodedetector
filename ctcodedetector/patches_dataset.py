import io3d
from .patches3d import  insert_patch_into_volume, make_patch
from matplotlib import pyplot as plt
import numpy as np
import skimage.exposure
import datetime
from pathlib import Path
import json
import skimage.io
from loguru import logger
from typing import Optional


def make_dataset(outputdir:Path, dataset_type="train", dataset_image_number:int=9, repeat_number:int=5, pilsenpigs_base_path:Optional[Path]=None):
    
    coco = create_coco_header()

    if not pilsenpigs_base_path:
        pilsenpigs_base_path = io3d.joinp(f"biomedical/orig/pilsen_pigs/")
    else:
        pilsenpigs_base_path = Path(pilsenpigs_base_path)
    iimage = 0
    for irepeat in range(repeat_number):
        for datai in range(1, dataset_image_number):
            iimage += 1
            ct_fn = pilsenpigs_base_path / f"{dataset_type}/PP_{datai:04d}/PATIENT_DICOM/PP_{datai:04d}.mhd"
            
            logger.info(ct_fn.name)

            datap = io3d.read(ct_fn)
            voxelsize_mm_scalar = .1

            # patch_size_px = [200, 200]
            patch_size_mm = [40, 40]
            background_density_hu = -1024

            # print('ahoj')
            patch = make_patch(
                voxelsize_mm_scalar, patch_size_mm,
                background_density_hu=background_density_hu,
                patch_object_density_hu=1500,
                patch_object_center_density_hu=800
            )

            # plt.figure()
            # plt.imshow(patch)
            # plt.colorbar()

            datapc, slices, patch3dr = insert_patch_into_volume(
                datap, patch,
                colon_radius_mm=10,
                patch_thickness_mm=1.0,
                voxelsize_mm_scalar=.1,
                background_density_hu=-1024
            )
            axis = 2
            im = np.max(datapc.data3d, axis=axis)
            im_rescal = skimage.exposure.rescale_intensity(im, in_range=(-100, 2000), out_range=(0, 255)).astype(np.uint8)
            # [top left x position, top left y position, width, height]

            removed = slices.pop(axis)

            bbox = list(map(int, [
                slices[1].start,
                slices[0].start,
                slices[1].stop - slices[1].start,
                slices[0].stop - slices[0].start,
            ]))


            # plt.figure(figsize=(15, 20))
            # plt.imshow(im_rescal, cmap='gray')
            # plt.colorbar()
            # plt.show()
            fn = outputdir / f"images_{dataset_type}" / f"{iimage:05d}.jpg"


            coco_image = {
                "license": 0,
                "file_name": str(fn.relative_to(outputdir)).replace("\\", "/"),
                "coco_url": "", #"http://images.cocodataset.org/val2017/000000397133.jpg",
                "height": im_rescal.shape[0],
                "width": im_rescal.shape[1],
                "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),#"2013-11-14 17:02:52",
                "flickr_url": "", #"http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
                "id": iimage
            }

            #bbox [top left x position, top left y position, width, height].
            coco_annotation = {
            "segmentation": [],
            "area": int(bbox[2] * bbox[3]),
            "iscrowd": 0,
            "image_id": iimage,
            "bbox": bbox,
            "category_id": 1,
            "id": iimage
            }

            coco["images"].append(coco_image)
            coco['annotations'].append(coco_annotation)

            fn.parent.mkdir(exist_ok=True, parents=True)
            skimage.io.imsave(fn, im_rescal)

        with open(outputdir / f"coco_{dataset_type}.json", "w") as f:
            json.dump(coco, f, indent=4)


def create_coco_header():
    coco = {
        "info" :{
            "description": "Pilsen Pigs with patch",
            "url": "",
            "version": "1.0",
            "year": 2022,
            "contributor": "UWB, FAV + CUNI, LFP",
            "date_created": "2022/08/30"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "patch","id": 1,"name": "patch_on"},
#             {"supercategory": "patch","id": 2,"name": "patch_off"},
            # {"supercategory": "vehicle","id": 3,"name": "car"},
            # {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
            # {"supercategory": "vehicle","id": 5,"name": "airplane"},
            # {"supercategory": "indoor","id": 89,"name": "hair drier"},
            # {"supercategory": "indoor","id": 90,"name": "toothbrush"}
        ],
        "segment_info": []
    }

    return coco