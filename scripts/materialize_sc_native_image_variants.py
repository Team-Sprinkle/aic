#!/usr/bin/env python3
"""Materialize a paired SC pose-label file using the collector's low-res RGB."""

import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_labels", type=Path)
    parser.add_argument("lowres_labels", type=Path)
    parser.add_argument("--oracle-crop-labels", type=Path,
                        help="Privileged geometry-selected crops for diagnostic upper bound only")
    parser.add_argument("--crop-size", type=int, default=224)
    args = parser.parse_args()
    args.lowres_labels.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    oracle = args.oracle_crop_labels.open("w") if args.oracle_crop_labels else None
    if oracle:
        crop_root = args.oracle_crop_labels.parent / "oracle_geometry_crops"
        crop_root.mkdir(parents=True, exist_ok=True)
    with args.native_labels.open() as src, args.lowres_labels.open("w") as dst:
        for line in src:
            row = json.loads(line)
            if oracle:
                diagnostic = {**row, "images": dict(row["images"])}
                for camera, pixels in row["projected_training_pixels"].items():
                    plug, port = pixels["tip"], pixels["opening"]
                    if plug[0] is None or port[0] is None:
                        center = (576, 512)
                    else:
                        center = ((plug[0] + port[0]) / 2, (plug[1] + port[1]) / 2)
                    left = round(center[0] - args.crop_size / 2)
                    top = round(center[1] - args.crop_size / 2)
                    target = crop_root / f"{row['trial']}_{row['frame']:06d}_{camera}.jpg"
                    with Image.open(row["images"][camera]) as im:
                        im.convert("RGB").crop((left, top, left + args.crop_size,
                                                top + args.crop_size)).save(target, quality=93)
                    diagnostic["images"][camera] = str(target)
                diagnostic.pop("projected_training_pixels")
                diagnostic["crop_selection"] = "ORACLE PRIVILEGED GEOMETRY; diagnostic upper bound only"
                oracle.write(json.dumps(diagnostic, separators=(",", ":")) + "\n")
            if not all(Path(path).is_file() for path in row["lowres_images"].values()):
                raise FileNotFoundError(row["lowres_images"])
            row["images"] = row.pop("lowres_images")
            row.pop("projected_training_pixels")
            dst.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    if oracle:
        oracle.close()
    print(json.dumps({"lowres_rows": count, "output": str(args.lowres_labels)}))


if __name__ == "__main__":
    main()
