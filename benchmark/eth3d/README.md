# ETH3D dataset

## Install

Using uv
```sh
uv sync
```

Or dependencies:
```sh
uv pip install .
```

Usage:
```sh
# download single dataset
# if dataset already exists in `datasets` it won't redownload
python3 -m eth3d courtyard -o datasets 

# download multiple dataset
python3 -m eth3d courtyard delivery_area -o datasets

# rescale dataset to new width, if dataset does not exist
# it will download first
python3 -m eth3d courtyard --width 2300 -o datasets

# download original jpg images next to the undistorted ones
python3 -m eth3d courtyard -o datasets --with-jpg

# download ground truth depth
python3 -m eth3d courtyard -o datasets --with-depth
```

we are downloading and extracting the following archives:
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

List of datasets:
- courtyard
- delivery_area
- electro
- facade
- kicker
- meadow
- office
- pipes
- playground
- relief
- relief_2
- terrace
- terrains
- botanical_garden
- boulders
- bridge
- door
- exhibition_hall
- lecture_room
- living_room
- lounge
- observatory
- old_computer
- statue
- terrace_2
