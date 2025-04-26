import os
import argparse

from eth3d.downloader import download_dataset
from eth3d.rescale import rescale_dataset

ETH3D_DATASETS_TRAINING = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
]

# datasets for which we don't have evaluation
ETH3D_DATASETS_TEST = [
    "botanical_garden",
    "boulders",
    "bridge",
    "door",
    "exhibition_hall",
    "lecture_room",
    "living_room",
    "lounge",
    "observatory",
    "old_computer",
    "statue",
    "terrace_2",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        "-o",
        required=True,
        type=str,
        help="Output directory to download the scans to.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Rescaling all images and cameras after downloading the original dataset. "
        "Rescale does not modify the original dataset, rather it creates "
        "a directory with the rescaled alternative.",
    )
    parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        choices=ETH3D_DATASETS_TRAINING + ETH3D_DATASETS_TEST,
        help="Any number of High-res dataset names from ETH3D to download.",
    )

    parser.add_argument(
        "--with-jpg",
        action="store_true",
        help="download original jpg images next to the undistorted ones",
    )
    parser.add_argument(
        "--with-depth", action="store_true", help="download ground truth depth as well"
    )

    return parser.parse_args()


def prepare_datasets(
    datasets: list[str],
    outdir: str,
    width: int | None = None,
    with_jpg: bool = False,
    with_depth: bool = False,
):

    for i, dataset in enumerate(datasets):
        if not os.path.exists(os.path.join(outdir, dataset)):
            print(f"{i+1} / {len(datasets)}: downloading {dataset}")
            download_dataset(
                dataset_name=dataset,
                datasets_dir=outdir,
                with_original_jpg=with_jpg,
                with_eval=(dataset in ETH3D_DATASETS_TRAINING),
                with_depth=(with_depth and (dataset in ETH3D_DATASETS_TRAINING)),
            )
        else:
            print(f"{i+1} / {len(datasets)}: {dataset} already exists, skipping.")

        rescaled_outpath = os.path.join(outdir, f"{dataset}_{width}")
        if width is not None and not os.path.exists(rescaled_outpath):
            print(f"{i+1} / {len(datasets)}: rescaling {dataset}")
            rescale_dataset(
                os.path.join(outdir, dataset),
                rescaled_outpath,
                new_width=width,
            )

def main():
    args = parse_args()
    prepare_datasets(
        args.datasets, args.outdir, args.width, args.with_jpg, args.with_depth
    )


# download all datasets
if __name__ == "__main__":
    main()
