"""
cydms.analyze
-------------
Main entry point — รับ MRI + EEG path คืน dict ผลลัพธ์ครบชุด
"""
from .mri import process_mri
from .eeg import load_eeg
from .source import compute_source_localization, compute_clinical_findings

DISCLAIMER = (
    "WARNING: This is a preliminary AI-based analysis only. "
    "Reference ranges are general estimates and may vary by age, condition, "
    "and recording equipment. Must be confirmed by a qualified physician."
)


def analyze(mri_path, eeg_path, tsv_elec_path=None, **kwargs):
    debug_log = []
    warnings  = []

    debug_log.append("🧠 Processing MRI...")
    mri_result = process_mri(mri_path)
    debug_log.append(f"🧠 MRI voxel size: {mri_result['voxel_size']:.2f}mm → threshold: {mri_result['threshold_pct']}%")

    debug_log.append("📡 Loading EEG...")
    raw, eeg_logs, eeg_warnings = load_eeg(eeg_path, tsv_elec_path=tsv_elec_path)
    debug_log.extend(eeg_logs)
    warnings.extend(eeg_warnings)

    debug_log.append("🔬 Computing source localization (sLORETA)...")
    results, waveform, src_logs = compute_source_localization(raw, debug_log=debug_log)
    debug_log.extend(src_logs)

    findings = compute_clinical_findings(
        results,
        mri_result['thickness'],
        mri_result['asymmetry']
    )

    return {
        'verts':             mri_result['verts'],
        'faces':             mri_result['faces'],
        'results':           results,
        'waveform':          waveform,
        'thickness':         mri_result['thickness'],
        'asymmetry':         mri_result['asymmetry'],
        'clinical_findings': findings,
        'warnings':          warnings,
        'debug_log':         debug_log,
        'disclaimer':        DISCLAIMER,
    }
