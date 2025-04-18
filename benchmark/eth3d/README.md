# ETH3D dataset

we are downloading and extracting:
- undistorted
- eval

In the following two sections, we are briefly describing the content of these two archives. To read more, please visit the official documentation of the dataset at: [https://www.eth3d.net/documentation](https://www.eth3d.net/documentation)

## Undistorted

Content:
- dslr_calibration_undistorted (COLMAP sparse reconstruction output)
    - points3D.txt
    - images.txt
    - cameras.txt
- images
    - dslr_images_undistorted
        - DSC_xxxx.JPG (`xxxx` id of the frame)
        - ...

## Eval

Content:
- scan1.ply
- scan2.ply
- ...
- scan_alignment.mlp

If more laser scans are available in a dataset, than more `.ply` files are present in the content of the `eval` archive. Between these, a `scan_alignment.mlp` file stores the 4x4 transformation matrix.
