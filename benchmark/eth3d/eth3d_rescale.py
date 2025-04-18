import os
import shutil

import cv2

import colmapio

def is_image(fname) -> bool:
    IMAGE_EXTS = ".png", ".jpg", ".jpeg"
    return os.path.isfile(fname) and os.path.splitext(fname)[1].lower() in IMAGE_EXTS

def eth3d_rescale_dataset(undistorted_path: str, new_width: int):
    """
    This function grabs a downloaded eth3d dataset and rescales the images and
    the corresponding camera intrinsics to a lower resolution given the new width.

               |-- path/to/dataset
                  |-- dataset_name_dslr_scan_eval
    (  reads ->)  |-- dataset_name_dslr_undistorted ( == <undistorted_path> )
                     |-- dataset_name
                        |-- ...
    (creates ->)  |-- dataset_name_dslr_undistorted_<new_width>
    """
    undistorted_path_content = os.listdir(undistorted_path)
    # we expect only one folder to be present here
    if len(undistorted_path_content) != 1:
        raise Exception("Incorrect number of artifacts in provided path,"
                        "while only a single folder is expected:\n"
                        f"{undistorted_path_content}")
    dataset_name = undistorted_path_content[0]
    undistorted_inner_path = os.path.join(undistorted_path, dataset_name)
    if not os.path.isdir(undistorted_inner_path):
        raise NotADirectoryError(undistorted_inner_path)

    original_images_dir = os.path.join(undistorted_inner_path, "images")
    original_sparse_dir = os.path.join(undistorted_inner_path, "dslr_calibration_undistorted")

    undistorted_rescaled_path = f"{undistorted_path}_{new_width}"
    undistorted_rescaled_inner_path = os.path.join(undistorted_rescaled_path, dataset_name)
    rescaled_images_dir = os.path.join(undistorted_rescaled_inner_path, "images")
    rescaled_sparse_dir = os.path.join(undistorted_rescaled_inner_path, "sparse")
    os.makedirs(rescaled_images_dir, exist_ok=True)
    os.makedirs(rescaled_sparse_dir, exist_ok=True)

    # RESCALE SPARSE
    cameras = colmapio.read_cameras_text(os.path.join(original_sparse_dir, "cameras.txt"))
    for camera_id in cameras:
        if cameras[camera_id].model != "PINHOLE":
            raise TypeError("Camera model must be PINHOLE (undistorted)")
        scaler_x = new_width / cameras[camera_id].width
        new_height = int(scaler_x * cameras[camera_id].height)
        scaler_y = new_height / cameras[camera_id].height
        # rescaling fx, fy, cx, cy with correct scaler
        rescaled_params = cameras[camera_id].params * [scaler_x, scaler_y, scaler_x, scaler_y]
        cameras[camera_id] = colmapio.Camera(id=cameras[camera_id].id,
               model=cameras[camera_id].model,
               width=new_width,
               height=new_height,
               params=rescaled_params)
    colmapio.write_cameras_text(cameras, os.path.join(rescaled_sparse_dir, "cameras.txt"))
    # images.txt and point3D.txt are the same so just copy them over
    shutil.copyfile(os.path.join(original_sparse_dir, "images.txt"), os.path.join(rescaled_sparse_dir, "images.txt"))
    shutil.copyfile(os.path.join(original_sparse_dir, "points3D.txt"), os.path.join(rescaled_sparse_dir, "points3D.txt"))

    for dirpath, _, fnames in os.walk(original_images_dir):
        for fname in fnames:
            fpath = os.path.join(dirpath, fname)
            if is_image(fpath):
                img = cv2.imread(fpath)
                height, width = img.shape[:2]
                img = cv2.resize(img, (new_width, int(new_width / width * height)))
                image_outpath = os.path.join(rescaled_images_dir, os.path.relpath(fpath, original_images_dir))
                os.makedirs(os.path.dirname(image_outpath), exist_ok=True)
                cv2.imwrite(image_outpath, img)

