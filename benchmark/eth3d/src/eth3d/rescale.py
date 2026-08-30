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


# Quality 100 with 4:4:4 sampling (no chroma subsampling) keeps the generation
# loss of the extra JPEG encode negligible, so resolution stays the only
# variable that differs between scales. OpenCV warns if these are handed to a
# non-JPEG encoder, hence the extension check.
JPEG_WRITE_PARAMS = [
    cv2.IMWRITE_JPEG_QUALITY,
    100,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
]


def copy_rescale_image(from_path, to_path, new_width):
    img = cv2.imread(from_path)
    height, width = img.shape[:2]
    # INTER_AREA area-averages the source pixels, band-limiting before it
    # decimates; INTER_LINEAR only taps 2x2 and so aliases when downscaling.
    interpolation = cv2.INTER_AREA if new_width < width else cv2.INTER_CUBIC
    img = cv2.resize(img, (new_width, int(new_width / width * height)), interpolation=interpolation)
    is_jpeg = os.path.splitext(to_path)[1].lower() in (".jpg", ".jpeg")
    cv2.imwrite(to_path, img, JPEG_WRITE_PARAMS if is_jpeg else [])


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