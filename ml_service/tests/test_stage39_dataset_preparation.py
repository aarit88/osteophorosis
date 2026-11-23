from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi import HTTPException

from ml_service.app.services.dataset_preparation import (
    DatasetPreparationService, PreparationConfig, PreparationError,
)
from ml_service.app.api.v1.research_routes import preparation_router, get_authenticated_reviewer, _require_preparation_role


class StagedOnboarding:
    def __init__(self, source_id="SRC_STAGE39", version="v1", track="track_d"):
        self.row = {"source_id": source_id, "version": version, "track": track,
            "verification_status": "VERIFIED", "staging_status": "STAGED_FOR_STAGE33_INTAKE",
            "package": {"valid": True}}

    def source(self, source_id):
        if source_id != self.row["source_id"]: raise KeyError("Dataset source not found.")
        return self.row


def build_staged(root: Path, *, corrupt=False):
    package = root / "SRC_STAGE39" / "v1"
    (package / "images").mkdir(parents=True)
    records = []
    for idx, (patient, site) in enumerate([
        ("PT_SYNTH_A", "Site Alpha"), ("PT_SYNTH_A", "Site Beta"),
        ("PT_SYNTH_B", "Site Alpha"), ("PT_SYNTH_C", "Center2 protected"),
    ]):
        rel = f"images/{idx}.png"
        if corrupt and idx == 0:
            (package / rel).write_bytes(b"not a real png")
        else:
            arr = np.random.default_rng(idx + 100).integers(0, 256, (16, 16), dtype=np.uint8)
            Image.fromarray(arr).save(package / rel)
        records.append({"sample_id": f"SM_SYNTH_{idx}", "patient_id": patient,
            "diagnostic_class": "paget", "imaging_modality": "xray", "image_path": rel,
            "institution_id": site, "dataset_source": "SRC_STAGE39"})
    manifest = {"manifest_version": "2.0.0", "track": "track_d", "records": records}
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    inventory = []
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        inventory.append({"path": path.relative_to(package).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (package / "provenance.json").write_text(json.dumps({"file_inventory": inventory}), encoding="utf-8")
    return package


def build_single_staged(root: Path, file_name: str, modality: str, *, patient_name: str | None = None):
    package = root / "SRC_STAGE39" / "v1"; package.mkdir(parents=True)
    record = {"sample_id": "SM_SYNTH_CT", "patient_id": "PT_SYNTH_CT", "diagnostic_class": "paget",
        "imaging_modality": modality, "image_path": file_name, "institution_id": "Site Fixture", "dataset_source": "SRC_STAGE39"}
    (package / "manifest.json").write_text(json.dumps({"manifest_version": "2.0.0", "track": "track_d", "records": [record]}), encoding="utf-8")
    if file_name.endswith(".dcm"):
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
        meta=FileMetaDataset(); meta.MediaStorageSOPClassUID=CTImageStorage; meta.MediaStorageSOPInstanceUID=generate_uid(); meta.TransferSyntaxUID=ExplicitVRLittleEndian
        ds=FileDataset(str(package/file_name),{},file_meta=meta,preamble=b"\0"*128)
        ds.SOPClassUID=CTImageStorage; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID; ds.Modality="CT"
        ds.Rows=16; ds.Columns=16; ds.SamplesPerPixel=1; ds.PhotometricInterpretation="MONOCHROME2"
        ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=1
        ds.RescaleType="HU"; ds.RescaleSlope="1"; ds.RescaleIntercept="-1000"
        ds.PixelSpacing=["0.5","0.5"]; ds.ImageOrientationPatient=["1","0","0","0","1","0"]
        if patient_name: ds.PatientName=patient_name
        ds.PixelData=np.arange(256,dtype=np.int16).tobytes(); ds.save_as(package/file_name)
    else:
        import nibabel as nib
        nib.save(nib.Nifti1Image(np.arange(16*16*4,dtype=np.float32).reshape(16,16,4),np.eye(4)),str(package/file_name))
    inventory=[]
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        inventory.append({"path":path.relative_to(package).as_posix(),"size_bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (package/"provenance.json").write_text(json.dumps({"file_inventory":inventory}),encoding="utf-8")
    return package


@pytest.fixture
def prepared(tmp_path):
    staged = tmp_path / "staged"
    build_staged(staged)
    service = DatasetPreparationService(tmp_path / "registry.json", staged, tmp_path / "out", tmp_path,
        onboarding=StagedOnboarding())
    run = service.create("SRC_STAGE39", "REV_DATA_STEWARD", PreparationConfig())
    return service, run["run_id"], staged


def test_inventory_is_deterministic_safe_and_patient_pseudonymous(prepared):
    service, run_id, _ = prepared
    run = service.inventory(run_id, "REV_DATA_STEWARD")
    assert run["state"] == "INVENTORY_COMPLETED"
    inventory = run["inventory"]
    assert inventory["file_count"] == 4
    refs = [row["patient_group_ref"] for row in inventory["groups"]]
    assert refs[0] == refs[1]
    serialized = json.dumps(inventory)
    assert "PT_SYNTH_A" not in serialized and "Site Alpha" not in serialized
    digest = inventory["inventory_sha256"]
    assert service.get(run_id)["inventory"]["inventory_sha256"] == digest
    assert run["lifecycle_events"][-1]["state"] == "INVENTORY_COMPLETED"
    second=service.create("SRC_STAGE39","REV_DATA_STEWARD",PreparationConfig())
    assert service.inventory(second["run_id"],"REV_DATA_STEWARD")["inventory"]["inventory_sha256"]==digest


def test_qc_blocks_corrupt_image_and_does_not_advance(tmp_path):
    staged = tmp_path / "staged"
    build_staged(staged, corrupt=True)
    service = DatasetPreparationService(tmp_path / "registry.json", staged, tmp_path / "out", tmp_path,
        onboarding=StagedOnboarding())
    run_id = service.create("SRC_STAGE39", "REV_DATA_STEWARD", PreparationConfig())["run_id"]
    service.inventory(run_id, "REV_DATA_STEWARD")
    result = service.qc(run_id, "REV_DATA_STEWARD")
    assert result["state"] == "BLOCKED"
    assert result["qc"]["blocker_count"] > 0
    with pytest.raises(PreparationError): service.harmonize(run_id, "REV_DATA_STEWARD")


def test_harmonization_is_separate_and_splits_exclude_protected_holdouts(prepared):
    service, run_id, _ = prepared
    service.inventory(run_id, "REV_DATA_STEWARD")
    assert service.qc(run_id, "REV_DATA_STEWARD")["state"] == "QC_COMPLETED"
    run = service.harmonize(run_id, "REV_DATA_STEWARD")
    assert run["state"] == "HARMONIZATION_COMPLETED"
    assert run["harmonization"]["source_modified"] is False
    assert run["harmonization"]["file_count"] == 4
    complete = service.prepare_splits(run_id, "REV_DATA_STEWARD")
    assert complete["state"] == "PREPARATION_COMPLETED"
    split = complete["splits"]
    assert split["excluded_protected_holdout_records"] == 1
    assert split["protected_holdouts_used_for_split"] is False
    assert split["leakage_audit"]["is_clean"] is True
    group_splits = {}
    for row in split["assignments"]:
        assert row["patient_group_ref"] not in group_splits or group_splits[row["patient_group_ref"]] == row["split"]
        group_splits[row["patient_group_ref"]] = row["split"]
    assert split["institution_overlap"]
    report = service.report(run_id)
    assert report["training_authorized"] is False
    assert report["clinical_readiness"] == "NOT_ESTABLISHED"
    assert "PT_SYNTH_A" not in json.dumps(report)
    report_file = service.output_root / run_id / "report.json"
    tampered = json.loads(report_file.read_text())
    tampered["clinical_validity"] = "VALID"
    report_file.write_text(json.dumps(tampered))
    with pytest.raises(PreparationError, match="report integrity"):
        service.report(run_id)

    repeat=service.create("SRC_STAGE39","REV_DATA_STEWARD",PreparationConfig())
    service.inventory(repeat["run_id"],"REV_DATA_STEWARD"); service.qc(repeat["run_id"],"REV_DATA_STEWARD")
    repeated=service.harmonize(repeat["run_id"],"REV_DATA_STEWARD")
    assert [x["sha256"] for x in repeated["harmonization"]["files"]] == [x["sha256"] for x in run["harmonization"]["files"]]
    repeated_splits=service.prepare_splits(repeat["run_id"],"REV_DATA_STEWARD")["splits"]["assignments"]
    assert repeated_splits == split["assignments"]


def test_staged_integrity_change_and_track_exclusion_block_run(tmp_path):
    staged = tmp_path / "staged"; package = build_staged(staged)
    service = DatasetPreparationService(tmp_path / "registry.json", staged, tmp_path / "out", tmp_path,
        onboarding=StagedOnboarding())
    (package / "images" / "0.png").write_bytes(b"changed")
    with pytest.raises(PreparationError, match="inventory"):
        service.create("SRC_STAGE39", "REV_DATA_STEWARD", PreparationConfig())
    track_c = DatasetPreparationService(tmp_path / "other-registry.json", staged, tmp_path / "out", tmp_path,
        onboarding=StagedOnboarding(track="track_c"))
    with pytest.raises(PreparationError, match="excludes Track A, B and C"):
        track_c.create("SRC_STAGE39", "REV_DATA_STEWARD", PreparationConfig())


def test_config_rejects_bad_ratios_and_routes_are_authenticated():
    with pytest.raises(ValueError): PreparationConfig(split={"train": .8, "validation": .3, "internal_test": .1, "seed": 1})
    routes = {route.path: route for route in preparation_router.routes}
    assert "/api/v1/dataset-preparation/runs" in routes
    for route in preparation_router.routes:
        assert any(dep.call is get_authenticated_reviewer for dep in route.dependant.dependencies)
    with pytest.raises(HTTPException) as denied:
        _require_preparation_role({"roles":frozenset({"CLINICAL_ADJUDICATOR"})})
    assert denied.value.status_code == 403


def test_dicom_privacy_qc_blocks_identifier_and_valid_hu_is_harmonized(tmp_path):
    pytest.importorskip("pydicom")
    staged=tmp_path/"staged"; build_single_staged(staged,"slice.dcm","ct",patient_name="SYNTHETIC^PATIENT")
    service=DatasetPreparationService(tmp_path/"registry.json",staged,tmp_path/"out",tmp_path,onboarding=StagedOnboarding())
    run=service.create("SRC_STAGE39","REV_DATA_STEWARD",PreparationConfig())
    service.inventory(run["run_id"],"REV_DATA_STEWARD")
    blocked=service.qc(run["run_id"],"REV_DATA_STEWARD")
    assert blocked["state"] == "BLOCKED"
    finding=next(row for row in blocked["qc"]["findings"] if row["rule_id"]=="DICOM_PHYSICAL_PRIVACY")
    assert "PatientName" in finding["evidence"]["tag_names"]
    assert "SYNTHETIC^PATIENT" not in json.dumps(blocked)

    staged2=tmp_path/"staged-valid"; build_single_staged(staged2,"slice.dcm","ct")
    service2=DatasetPreparationService(tmp_path/"registry2.json",staged2,tmp_path/"out2",tmp_path,onboarding=StagedOnboarding())
    run2=service2.create("SRC_STAGE39","REV_DATA_STEWARD",PreparationConfig())
    service2.inventory(run2["run_id"],"REV_DATA_STEWARD")
    assert service2.qc(run2["run_id"],"REV_DATA_STEWARD")["state"]=="QC_COMPLETED"
    result=service2.harmonize(run2["run_id"],"REV_DATA_STEWARD")
    assert result["harmonization"]["files"][0]["window_hu"]=={"center":400.0,"width":1500.0}


def test_nifti_ct_requires_documented_hu_evidence(tmp_path):
    pytest.importorskip("nibabel")
    staged=tmp_path/"staged"; build_single_staged(staged,"volume.nii.gz","ct")
    service=DatasetPreparationService(tmp_path/"registry.json",staged,tmp_path/"out",tmp_path,onboarding=StagedOnboarding())
    default=PreparationConfig()
    run=service.create("SRC_STAGE39","REV_DATA_STEWARD",default); service.inventory(run["run_id"],"REV_DATA_STEWARD")
    service.qc(run["run_id"],"REV_DATA_STEWARD")
    with pytest.raises(PreparationError,match="evidence reference"):
        service.harmonize(run["run_id"],"REV_DATA_STEWARD")
    staged2=tmp_path/"staged-evidence"; build_single_staged(staged2,"volume.nii.gz","ct")
    service2=DatasetPreparationService(tmp_path/"registry2.json",staged2,tmp_path/"out2",tmp_path,onboarding=StagedOnboarding())
    config=PreparationConfig(ct={"window_center":400,"window_width":1500,"require_validated_hu":True,"nifti_hu_evidence_reference":"source-document-reference"})
    run2=service2.create("SRC_STAGE39","REV_DATA_STEWARD",config); service2.inventory(run2["run_id"],"REV_DATA_STEWARD")
    assert "source-document-reference" not in json.dumps(run2)
    service2.qc(run2["run_id"],"REV_DATA_STEWARD")
    assert service2.harmonize(run2["run_id"],"REV_DATA_STEWARD")["harmonization"]["file_count"]==1
