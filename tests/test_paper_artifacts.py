from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]


def test_figure_manifest_binds_every_input_and_output() -> None:
    figure_root = ROOT / "paper" / "figures"
    manifest = json.loads((figure_root / "figure_manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert len(manifest["inputs"]) == 9
    assert len(manifest["outputs"]) == 12
    assert set(manifest["outputs"]) == set(manifest["output_sha256"])
    for record in manifest["inputs"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert _digest(path) == record["sha256"]
    for name, expected in manifest["output_sha256"].items():
        path = figure_root / name
        assert path.is_file()
        assert _digest(path) == expected
        assert path.stat().st_size > 1_000


def test_figure_pngs_have_publication_resolution_and_matching_captions() -> None:
    figure_root = ROOT / "paper" / "figures"
    manifest = json.loads((figure_root / "figure_manifest.json").read_text())
    pngs = [name for name in manifest["outputs"] if name.endswith(".png")]
    assert len(pngs) == 6
    for name in pngs:
        with Image.open(figure_root / name) as image:
            assert image.width >= 2_000
            assert image.height >= 800
    captions = (ROOT / "paper" / "figure_captions.md").read_text()
    assert re.findall(r"^## Figure (\d+):", captions, flags=re.MULTILINE) == [
        str(index) for index in range(1, 7)
    ]


def test_versioned_result_manifests_bind_their_artifacts() -> None:
    manifests = sorted((ROOT / "paper" / "results").glob("*/artifact_manifest.json"))
    assert len(manifests) >= 3
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        assert re.fullmatch(r"[0-9a-f]{40}", manifest["analysis_code_commit"])
        for name, expected in manifest["artifacts"].items():
            path = manifest_path.parent / name
            assert path.is_file()
            assert _digest(path) == expected


def test_every_manuscript_citation_resolves() -> None:
    manuscript = (ROOT / "paper" / "manuscript.md").read_text()
    bibliography = (ROOT / "references" / "references.bib").read_text()
    cited = set(re.findall(r"@([A-Za-z0-9_:-]+)", manuscript))
    available = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert len(cited) >= 18
    assert cited <= available


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
