"""Experiment specification: parse, validate, expand into the run list.

R-EXP-01..R-EXP-04, R-RUN-01, R-RUN-08, R-STA-01, R-STA-03.
"""

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import SCHEMA_VERSION

DEFAULT_TIMEOUT_S = 6 * 3600
DEFAULT_IMAGE = "sota-deps:latest"
DEFAULT_TOLERANCES = (0.01, 0.02, 0.05, 0.1, 0.2)
DEFAULT_PRIMARY_TOLERANCE = 0.02

DEFAULT_LAYOUT = {
    # Relative to `paths.dataset_root`. `{scene}` and `{width}` are the only
    # placeholders; a campaign that runs at native resolution needs its own
    # template because ETH3D scene widths differ.
    "scene_dir": "{scene}_{width}",
    "raw_scene_dir": "{scene}_dslr_undistorted/{scene}",
    "ground_truth_mlp": "{scene}_dslr_scan_eval/{scene}/dslr_scan_eval/scan_alignment.mlp",
}

DEFAULT_CONTAINER_PATHS = {
    "method_root": "/sota",
    "python": "/sota/.venv/bin/python",
    "data": "/data",
    "work": "/work",
    "out": "/out",
    "evaluator": "ETH3DMultiViewEvaluation",
}

# Argument lists the harness appends to the shared converter invocation, per
# configuration name; a spec may override this map.
DEFAULT_CONVERTER_ARGS = {
    "author": [],
    "norm10": ["--neighbours", "10"],
}

# CUMVS reads COLMAP output through its own initialiser, so the neighbour count
# of a normalized configuration reaches it there (methods.yaml, CUMVS
# harness_deviations).
DEFAULT_INITIALIZER_ARGS = {
    "author": [],
    "norm10": ["--max-neighbors=10"],
}


class SpecError(Exception):
    """The specification cannot be executed as written."""


@dataclass(frozen=True)
class Run:
    method: str
    scene: str
    width: int
    configuration: str
    repeat: int

    @property
    def key(self):
        return (
            f"{self.method}__{self.scene}__w{self.width}"
            f"__{self.configuration}__r{self.repeat}"
        )


@dataclass
class Spec:
    path: Path
    text: str
    data: dict
    campaign: str
    image: str
    methods: list
    scenes: list
    widths: list
    configurations: list
    repeats: int
    timeout_s: int
    exclusions: list
    order_seed: int
    shard_index: int
    shard_count: int
    gpu_index: int
    paths: dict
    layout: dict
    container: dict
    converter_args: dict
    initializer_args: dict
    method_env: dict
    phase_timer: bool
    debug_output: str
    padding: str
    tolerances: list
    primary_tolerance: float
    declared_fingerprint: dict
    evaluator_sha: str = None
    keep_intermediates: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def campaign_dir(self):
        return Path(self.paths["results_root"]) / self.campaign

    def scene_dir(self, scene, width):
        return Path(self.paths["dataset_root"]) / self.layout["scene_dir"].format(
            scene=scene, width=width
        )

    def raw_scene_dir(self, scene, width):
        return self.scene_dir(scene, width) / self.layout["raw_scene_dir"].format(
            scene=scene, width=width
        )

    def ground_truth_mlp(self, scene, width):
        return self.scene_dir(scene, width) / self.layout["ground_truth_mlp"].format(
            scene=scene, width=width
        )

    def work_dir(self, run):
        return Path(self.paths["work_root"]) / self.campaign / run.key

    def artifact_path(self, run):
        return Path(self.paths["artifacts_root"]) / self.campaign / f"{run.key}.ply"


def _require(data, key, where):
    if key not in data or data[key] is None:
        raise SpecError(f"{where}: required field `{key}` is missing")
    return data[key]


def _as_list(value, key):
    if not isinstance(value, list) or not value:
        raise SpecError(f"`{key}` must be a non-empty list")
    return value


def load(path, manifest=None):
    """Parse and validate an experiment specification.

    `manifest` is the parsed methods.yaml; when given, method and
    configuration names are checked against it and `status: excluded` methods
    are refused (R-EXP-01).
    """
    path = Path(path)
    text = path.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SpecError(f"{path}: the specification must be a YAML mapping")

    version = _require(data, "version", str(path))
    if version != SCHEMA_VERSION:
        raise SpecError(
            f"{path}: schema version {version} is not implemented "
            f"(this harness implements version {SCHEMA_VERSION})"
        )

    paths = dict(_require(data, "paths", str(path)))
    for key in ("dataset_root", "results_root", "work_root", "artifacts_root"):
        _require(paths, key, f"{path}: paths")
    # Container mounts need absolute paths; symlinks are left alone so that a
    # record shows the path the operator configured.
    paths = {k: str(Path(v).expanduser().absolute()) for k, v in paths.items()}

    layout = dict(DEFAULT_LAYOUT)
    layout.update(data.get("dataset_layout") or {})
    container = dict(DEFAULT_CONTAINER_PATHS)
    container.update(data.get("container_paths") or {})
    converter_args = dict(DEFAULT_CONVERTER_ARGS)
    converter_args.update(data.get("converter_args") or {})
    initializer_args = dict(DEFAULT_INITIALIZER_ARGS)
    initializer_args.update(data.get("initializer_args") or {})

    # R-EXP-11: `off` passes the shared --no-debug-output flag to every method
    # whose invocation carries it; `upstream` removes the token, so the method
    # writes the debug artefacts its authors shipped it writing. The pair is a
    # measured result, not only a disclosure (deviation policy rule 1).
    debug_output = data.get("debug_output", "off")
    if isinstance(debug_output, bool):
        # YAML 1.1 resolves a bare `off` to False, so an unquoted `debug_output:
        # off` never reaches this function as the string the operator wrote.
        raise SpecError(
            f"{path}: `debug_output` was parsed as the YAML boolean "
            f"{debug_output}; write it quoted, as `debug_output: \"off\"`"
        )
    debug_output = str(debug_output)
    if debug_output not in ("off", "upstream"):
        raise SpecError(
            f"{path}: `debug_output` must be `off` (R-EXP-11 in force) or "
            "`upstream` (the released behaviour)"
        )

    padding = str(data.get("padding", "none"))
    if padding not in ("none", "all"):
        raise SpecError(
            f"{path}: `padding` must be `none` or `all`; R-EXP-09 allows no "
            "per-method padding under a shared converter"
        )

    campaign = str(_require(data, "campaign", str(path)))
    if "/" in campaign or campaign.startswith("."):
        raise SpecError(f"{path}: `campaign` must be a plain directory name")

    spec = Spec(
        path=path,
        text=text,
        data=data,
        campaign=campaign,
        image=str(data.get("image") or DEFAULT_IMAGE),
        methods=_as_list(_require(data, "methods", str(path)), "methods"),
        scenes=_as_list(_require(data, "scenes", str(path)), "scenes"),
        widths=[int(w) for w in _as_list(_require(data, "widths", str(path)), "widths")],
        configurations=_as_list(
            _require(data, "configurations", str(path)), "configurations"
        ),
        repeats=int(_require(data, "repeats", str(path))),
        timeout_s=int(data.get("timeout_seconds") or DEFAULT_TIMEOUT_S),
        exclusions=list(data.get("exclude") or []),
        # A campaign with no declared seed still has to be reproducible, so the
        # order is drawn from the campaign name rather than from the clock.
        order_seed=int(
            data.get("order_seed")
            if data.get("order_seed") is not None
            else int(hashlib.blake2b(campaign.encode(), digest_size=4).hexdigest(), 16)
        ),
        shard_index=int((data.get("shard") or {}).get("index", 0)),
        shard_count=int((data.get("shard") or {}).get("count", 1)),
        gpu_index=int(data.get("gpu_index", 0)),
        paths=paths,
        layout=layout,
        container=container,
        converter_args=converter_args,
        initializer_args=initializer_args,
        method_env=dict(data.get("method_env") or {}),
        phase_timer=bool(data.get("phase_timer", True)),
        debug_output=debug_output,
        padding=padding,
        tolerances=[float(t) for t in (data.get("tolerances") or DEFAULT_TOLERANCES)],
        primary_tolerance=float(
            data.get("primary_tolerance", DEFAULT_PRIMARY_TOLERANCE)
        ),
        declared_fingerprint=dict(data.get("fingerprint") or {}),
        evaluator_sha=data.get("evaluator_sha"),
        keep_intermediates=bool(data.get("keep_intermediates", False)),
        extra={
            k: v
            for k, v in data.items()
            if k
            not in {
                "version",
                "campaign",
                "image",
                "methods",
                "scenes",
                "widths",
                "configurations",
                "repeats",
                "timeout_seconds",
                "exclude",
                "order_seed",
                "shard",
                "gpu_index",
                "paths",
                "dataset_layout",
                "container_paths",
                "converter_args",
                "initializer_args",
                "method_env",
                "phase_timer",
                "debug_output",
                "padding",
                "tolerances",
                "primary_tolerance",
                "fingerprint",
                "evaluator_sha",
                "keep_intermediates",
            }
        },
    )

    if spec.repeats < 1:
        raise SpecError(f"{path}: `repeats` must be at least 1 (R-STA-03)")
    if spec.shard_count < 1 or not (0 <= spec.shard_index < spec.shard_count):
        raise SpecError(f"{path}: `shard` must satisfy 0 <= index < count")
    if spec.primary_tolerance not in spec.tolerances:
        raise SpecError(
            f"{path}: `primary_tolerance` {spec.primary_tolerance} is not in `tolerances`"
        )

    if manifest is not None:
        validate_against_manifest(spec, manifest)
    return spec


def method_entry(manifest, name):
    for entry in manifest.get("methods") or []:
        if entry.get("name") == name:
            return entry
    return None


def validate_against_manifest(spec, manifest):
    """R-EXP-01: unknown or excluded methods, unknown configurations."""
    known_configs = set(manifest.get("configurations") or {})
    for name in spec.configurations:
        if name not in known_configs:
            raise SpecError(
                f"configuration `{name}` is not defined in methods.yaml "
                f"(known: {', '.join(sorted(known_configs))})"
            )
    for name in spec.methods:
        entry = method_entry(manifest, name)
        if entry is None:
            raise SpecError(f"method `{name}` is not in methods.yaml")
        if entry.get("status") == "excluded":
            reason = " ".join(str(entry.get("reason", "")).split())
            raise SpecError(
                f"method `{name}` is excluded from the campaign by methods.yaml "
                f"and may not be measured. Reason: {reason}"
            )
        # R-EXP-09: under a shared converter, padding is applied to every
        # method or to none, so a method that enforces equal image sizes
        # forces the whole batch's choice and the operator has to make it.
        shared = [c for c in spec.configurations if c != "author"]
        declared_equal = bool(spec.data.get("scenes_have_equal_image_sizes", False))
        if entry.get("requires_equal_image_sizes") and shared and spec.padding != "all":
            for scene in spec.scenes:
                for width in spec.widths:
                    sizes = scene_image_sizes(spec.raw_scene_dir(scene, width))
                    if sizes is None and not declared_equal:
                        raise SpecError(
                            f"method `{name}` aborts unless every image has the "
                            f"same size, configuration(s) {', '.join(shared)} use "
                            f"the shared converter, and scene `{scene}` is not on "
                            "disk to check. Either set `padding: all`, or assert "
                            "`scenes_have_equal_image_sizes: true` (R-EXP-09)."
                        )
                    if sizes is not None and len(sizes) > 1:
                        dims = ", ".join(f"{w}x{h}:{n}" for (w, h), n in sizes.items())
                        raise SpecError(
                            f"scene `{scene}` at width {width} has {len(sizes)} image "
                            f"sizes ({dims}); method `{name}` would abort and "
                            "R-EXP-09 forbids per-method padding. Choose a uniform "
                            "scene (F-021) or set `padding: all`."
                        )


def scene_image_sizes(raw_scene_dir):
    """Distinct (width, height) → image count from the scene's COLMAP text
    calibration, or None when the scene is not on disk. The `python -m eth3d
    --width` rescale rewrites cameras.txt, so this reads the sizes the methods
    will actually see."""
    cal = Path(raw_scene_dir) / "dslr_calibration_undistorted"
    if not (cal / "cameras.txt").is_file() or not (cal / "images.txt").is_file():
        return None
    cams = {}
    for line in (cal / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        cams[f[0]] = (int(f[2]), int(f[3]))
    counts = {}
    for line in (cal / "images.txt").read_text().splitlines():
        f = line.split()
        # An image row has exactly ten fields; the POINTS2D row that follows it
        # has a multiple of three (possibly zero).
        if len(f) == 10 and not line.startswith("#"):
            size = cams[f[8]]
            counts[size] = counts.get(size, 0) + 1
    return counts


def _matches(run, rule):
    """An exclusion rule matches a run when every field it names matches."""
    fields = {
        "method": run.method,
        "scene": run.scene,
        "width": run.width,
        "configuration": run.configuration,
        "config": run.configuration,
        "repeat": run.repeat,
    }
    unknown = set(rule) - set(fields)
    if unknown:
        raise SpecError(f"exclusion rule has unknown field(s): {', '.join(sorted(unknown))}")
    return all(fields[k] == v for k, v in rule.items())


def shard_of(key, shard_count):
    """Deterministic shard for a run key (R-RUN-08)."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % shard_count


def expand(spec):
    """The run list: Cartesian product minus exclusions, randomised, sharded.

    Order is randomised across repeats (R-STA-01) with the spec's `order_seed`,
    so the same specification always produces the same order and the order is
    recorded in every run record.
    """
    runs = []
    for repeat in range(1, spec.repeats + 1):
        for method in spec.methods:
            for scene in spec.scenes:
                for width in spec.widths:
                    for configuration in spec.configurations:
                        run = Run(method, scene, width, configuration, repeat)
                        if any(_matches(run, rule) for rule in spec.exclusions):
                            continue
                        runs.append(run)
    random.Random(spec.order_seed).shuffle(runs)
    if spec.shard_count > 1:
        runs = [r for r in runs if shard_of(r.key, spec.shard_count) == spec.shard_index]
    return runs
