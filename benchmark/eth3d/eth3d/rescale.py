import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import tqdm

import eth3d.colmapio as colmapio


def is_image(fname) -> bool:
    IMAGE_EXTS = ".png", ".jpg", ".jpeg"
    return os.path.isfile(fname) and os.path.splitext(fname)[1].lower() in IMAGE_EXTS


def copy_rescale_cameras_txt(from_path, to_path, new_width):
    cameras = colmapio.read_cameras_text(from_path)
    for camera_id in cameras:
        if cameras[camera_id].model != "PINHOLE":
            raise TypeError("Camera model must be PINHOLE (undistorted)")
        scaler_x = new_width / cameras[camera_id].width
        new_height = int(scaler_x * cameras[camera_id].height)
        scaler_y = new_height / cameras[camera_id].height
        # rescaling fx, fy, cx, cy with correct scaler
        rescaled_params = cameras[camera_id].params * [
            scaler_x,
            scaler_y,
            scaler_x,
            scaler_y,
        ]
        cameras[camera_id] = colmapio.Camera(
            id=cameras[camera_id].id,
            model=cameras[camera_id].model,
            width=new_width,
            height=new_height,
            params=rescaled_params,
        )
    colmapio.write_cameras_text(cameras, to_path)


def copy_rescale_image(from_path, to_path, new_width):
    img = cv2.imread(from_path)
    height, width = img.shape[:2]
    img = cv2.resize(img, (new_width, int(new_width / width * height)))
    cv2.imwrite(to_path, img)


def copy_other(from_path, to_path, _):
    shutil.copyfile(from_path, to_path)


def rescale_dataset(dataset_inpath: str, rescaled_outpath: str, new_width: int, multithreaded: bool = True):
    """
    This function grabs a downloaded eth3d dataset and rescales the images and
    the corresponding camera intrinsics to a lower resolution given the new width.
    """
    def copy_file(from_path, to_path):
        os.makedirs(os.path.dirname(os.path.abspath(to_path)), exist_ok=True)
        match os.path.basename(from_path):
            case "cameras.txt":
                copy_rescale_cameras_txt(from_path, to_path, new_width)
            case _ if is_image(from_path):
                copy_rescale_image(from_path, to_path, new_width)
            case _:
                copy_other(from_path, to_path, new_width)

    from_paths = [os.path.join(dirpath, fname) for dirpath, _, fnames in os.walk(dataset_inpath) for fname in fnames]
    to_paths = [os.path.join(rescaled_outpath, os.path.relpath(from_path, dataset_inpath)) for from_path in from_paths]
    pbar = tqdm.tqdm(total=len(from_paths), desc="Rescaling files")

    if multithreaded:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for _ in executor.map(copy_file, from_paths, to_paths):
                pbar.update()
    else:
        for from_path, to_path in zip(from_paths, to_paths):
            copy_file(from_path, to_path)
            pbar.update()