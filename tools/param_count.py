"""Report trainable parameter counts and output shapes for model YAML configs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import yaml

from model.builder import build_model_from_yaml

_DUMMY_SHAPES = {"1d": (2, 1, 4096), "2d": (2, 1, 64, 64)}


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print trainable param counts and output shapes for model YAML configs."
    )
    parser.add_argument("configs", nargs="+", help="Paths to model YAML files.")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes.")
    args = parser.parse_args()

    rows = []
    failed = False

    for path_str in args.configs:
        path = Path(path_str)
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            dim_type = cfg.get("type", "1d")
            if dim_type not in _DUMMY_SHAPES:
                raise ValueError(f"type must be '1d' or '2d', got '{dim_type}'")
            model = build_model_from_yaml(path, num_classes=args.num_classes)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(*_DUMMY_SHAPES[dim_type]))
            rows.append((cfg.get("name", path.stem), _count_params(model), tuple(out.shape)))
        except Exception as e:
            print(f"FAILED: {path} -> {e}", file=sys.stderr)
            failed = True

    if rows:
        name_w = max(len("config"), max(len(r[0]) for r in rows))
        params_w = max(len("params"), max(len(str(r[1])) for r in rows))
        shape_w = max(len("output_shape"), max(len(str(r[2])) for r in rows))
        header = f"{'config':<{name_w}}  {'params':>{params_w}}  {'output_shape':<{shape_w}}"
        print(header)
        print("-" * len(header))
        for name, params, shape in rows:
            print(f"{name:<{name_w}}  {params:>{params_w}}  {str(shape):<{shape_w}}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
