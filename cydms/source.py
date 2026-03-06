"""
cydms.source
------------
Forward solution, inverse solution (sLORETA), band power, clean segment selection
"""
import numpy as np
import mne


BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 45),
}

COLORS = {
    'Delta': '#ff3333',
    'Theta': '#ffcc00',
    'Alpha': '#33ff33',
    'Beta':  '#3333ff',
    'Gamma': '#ff33ff',
}

NORMAL_RANGES = {
    'Delta': (5, 25),
    'Theta': (5, 25),
    'Alpha': (25, 45),
    'Beta':  (10, 30),
    'Gamma': (2, 15),
}

DIAG_MAP = {
    'Delta': {
        'high':   '⚠️ สูงกว่าปกติ: เสี่ยง ADHD, บาดเจ็บสมอง, หรือง่วงมาก',
        'normal': '✅ ปกติ: สมองพักผ่อนหรือนอนหลับลึก',
        'low':    '⚠️ ต่ำกว่าปกติ: อาจนอนหลับไม่พอ',
    },
    'Theta': {
        'high':   '⚠️ สูงกว่าปกติ: เสี่ยง ADHD, วิตกกังวล, หรือซึมเศร้า',
        'normal': '✅ ปกติ: ผ่อนคลาย ใกล้หลับ',
        'low':    '⚠️ ต่ำกว่าปกติ: ตื่นตัวสูงมากผิดปกติ',
    },
    'Alpha': {
        'high':   '✅ สูงกว่าปกติ: ผ่อนคลายดีเยี่ยม สมาธิดี',
        'normal': '✅ ปกติ: สมองทำงานสมดุล',
        'low':    '⚠️ ต่ำกว่าปกติ: เครียด วิตกกังวล หรือสมาธิลดลง',
    },
    'Beta': {
        'high':   '⚠️ สูงกว่าปกติ: เครียดสูง วิตกกังวล หรือ OCD',
        'normal': '✅ ปกติ: ตื่นตัว ทำงานได้ดี',
        'low':    '⚠️ ต่ำกว่าปกติ: ง่วงนอน หรือขาดสมาธิ',
    },
    'Gamma': {
        'high':   '⚠️ สูงกว่าปกติ: Hyperactivity หรือเครียดสูงมาก',
        'normal': '✅ ปกติ: ประมวลผลข้อมูลดี',
        'low':    '⚠️ ต่ำกว่าปกติ: การประมวลผลสมองช้าลง',
    },
}


def get_lobe_name(pos):
    y, z = pos[1], pos[2]
    if y > 25:   return "Frontal Lobe"
    elif y < -45: return "Occipital Lobe"
    elif z < -15: return "Temporal Lobe"
    else:         return "Parietal Lobe"


def find_clean_segment(band_data, sfreq, win_sec=25.0, step_sec=5.0, debug_log=None):
    n_samples = band_data.shape[1]
    win  = int(win_sec * sfreq)
    step = int(step_sec * sfreq)
    if n_samples <= win:
        return None, 'low'

    wins, variances = [], []
    for i in range(0, n_samples - win, step):
        seg = band_data[:, i:i+win]
        wins.append(i)
        variances.append(float(np.var(seg)))

    variances  = np.array(variances)
    median_var = np.median(variances)
    diffs      = np.abs(variances - median_var)
    best_idx   = np.argmin(diffs)

    if diffs[best_idx] > median_var * 2:
        if debug_log is not None:
            debug_log.append("⚠️ ไฟล์ noisy มาก — ใช้ทั้งไฟล์แทน")
        return None, 'low'

    start = wins[best_idx]
    return (start, start + win), 'normal'


def compute_source_localization(raw, debug_log=None):
    if debug_log is None:
        debug_log = []
    src_logs = []

    fs_dir      = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = __import__('os').path.dirname(fs_dir)
    src = mne.setup_source_space('fsaverage', spacing='oct2',
                                  subjects_dir=subjects_dir, add_dist=False, verbose=False)
    bem = __import__('os').path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    try:
        fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src,
                                         bem=bem, eeg=True, mindist=2.0,
                                         ignore_ref=False, verbose=False)
    except Exception as e_fwd:
        err_msg = str(e_fwd)
        if "No EEG channels" in err_msg:
            err_msg += " (Make sure your SET file contains EEG channels)"
        raise RuntimeError(f"Forward solution failed: {err_msg}")

    inv  = mne.minimum_norm.make_inverse_operator(raw.info, fwd,
                                                    mne.make_ad_hoc_cov(raw.info),
                                                    verbose=False)
    sfreq      = raw.info['sfreq']
    final_results = {}
    total_pwr  = 0
    raw_results = []

    for name, (fmin, fmax) in BANDS.items():
        raw_filtered = raw.copy().filter(fmin, fmax, verbose=False)
        band_data    = raw_filtered.get_data()

        seg_range, conf = find_clean_segment(band_data, sfreq, debug_log=src_logs)
        if seg_range:
            clean_data = band_data[:, seg_range[0]:seg_range[1]]
            src_logs.append(f"✅ {name}: ใช้ช่วง {seg_range[0]/sfreq:.1f}s-{seg_range[1]/sfreq:.1f}s")
        else:
            clean_data = band_data
            src_logs.append(f"{'⚠️' if conf=='low' else '✅'} {name}: ใช้ทั้งไฟล์ (confidence: {conf})")

        stc    = mne.minimum_norm.apply_inverse_raw(raw_filtered, inv,
                                                     lambda2=1.0/9.0,
                                                     method='sLORETA', verbose=False)
        d_mean = np.mean(np.abs(stc.data), axis=1)
        pk     = np.argmax(d_mean)
        n_lh   = len(src[0]['rr'])
        p_std  = src[0]['rr'][pk] if pk < n_lh else src[1]['rr'][pk - n_lh]

        pwr = float(np.mean(np.abs(clean_data)))
        total_pwr += pwr
        raw_results.append({
            'name': name,
            'pos': (p_std * 100 * 2.5).tolist(),
            'pos_raw': (p_std * 100).tolist(),
            'pwr': pwr,
            'confidence': conf,
        })

    for item in raw_results:
        pct  = round((item['pwr'] / (total_pwr if total_pwr > 0 else 1)) * 100, 2)
        lobe = get_lobe_name(item['pos_raw'])
        name = item['name']
        lo, hi = NORMAL_RANGES.get(name, (10, 30))
        status = 'high' if pct > hi else ('low' if pct < lo else 'normal')
        diag   = DIAG_MAP.get(name, {}).get(status, '')
        final_results[name] = {
            'val': pct,
            'pos': item['pos'],
            'color': COLORS[name],
            'lobe': lobe,
            'status': status,
            'diag': diag,
            'confidence': item.get('confidence', 'normal'),
        }

    # Waveform
    waveform = {}
    try:
        ds       = max(1, int(sfreq // 50))
        waveform['times'] = raw.times[::ds].tolist()
        for band_name, (fmin, fmax) in BANDS.items():
            band_data = raw.copy().filter(fmin, fmax, verbose=False).get_data()
            avg = np.mean(band_data, axis=0)[::ds]
            mx  = np.max(np.abs(avg)) or 1
            waveform[band_name] = (avg / mx).tolist()
    except:
        pass

    return final_results, waveform, src_logs


def compute_clinical_findings(results, thickness, asymmetry_pct):
    asymmetry = asymmetry_pct / 100.0

    delta_pct  = results.get('Delta', {}).get('val', 0)
    theta_pct  = results.get('Theta', {}).get('val', 0)
    alpha_pct  = results.get('Alpha', {}).get('val', 0)
    beta_pct   = results.get('Beta',  {}).get('val', 0)
    gamma_pct  = results.get('Gamma', {}).get('val', 0)

    delta_st = results.get('Delta', {}).get('status', 'normal')
    theta_st = results.get('Theta', {}).get('status', 'normal')
    alpha_st = results.get('Alpha', {}).get('status', 'normal')
    beta_st  = results.get('Beta',  {}).get('status', 'normal')
    gamma_st = results.get('Gamma', {}).get('status', 'normal')
    delta_lobe = results.get('Delta', {}).get('lobe', '')

    mean_thick      = np.mean(list(thickness.values()))
    mri_focal_thick = any(v > mean_thick * 1.3 for v in thickness.values())
    mri_focal_thin  = any(v < mean_thick * 0.7 for v in thickness.values())
    mri_asymmetry   = asymmetry > 0.08

    findings = []

    if delta_st == 'high' and mri_focal_thick:
        findings.append({
            'condition': 'Suspected Brain Tumor',
            'confidence': 'HIGH' if delta_pct > 30 else 'MEDIUM',
            'evidence': [f'Delta abnormally high ({delta_pct}%) at {delta_lobe}', 'MRI shows abnormal cortical thickness'],
            'recommend': 'Recommend MRI with contrast for further evaluation',
        })
    elif delta_st == 'high':
        findings.append({
            'condition': 'Possible Brain Tumor',
            'confidence': 'LOW',
            'evidence': [f'Delta abnormally high ({delta_pct}%) at {delta_lobe}', 'No clear MRI thickness abnormality'],
            'recommend': 'Follow up and further examination recommended',
        })

    if delta_st == 'high' and (mri_focal_thin or mri_asymmetry):
        findings.append({
            'condition': 'Suspected Epilepsy',
            'confidence': 'HIGH' if (mri_focal_thin and mri_asymmetry) else 'MEDIUM',
            'evidence': [f'Delta abnormally high ({delta_pct}%)',
                         'MRI shows cortical thinning or asymmetry' if mri_focal_thin else 'MRI shows asymmetry'],
            'recommend': 'Recommend prolonged EEG monitoring and neurology consult',
        })
    elif delta_st == 'high' and not mri_focal_thick:
        findings.append({
            'condition': 'Possible Epilepsy',
            'confidence': 'LOW',
            'evidence': [f'Delta abnormally high ({delta_pct}%)', 'No clear MRI abnormality'],
            'recommend': 'Monitor symptoms and consult a physician',
        })

    if theta_st == 'high' and alpha_st == 'low':
        findings.append({
            'condition': 'Suspected ADHD',
            'confidence': 'MEDIUM',
            'evidence': [f'Theta elevated ({theta_pct}%)', f'Beta elevated ({beta_pct}%)', f'Alpha low ({alpha_pct}%)'],
            'recommend': 'Recommend psychological assessment and physician consult',
        })

    if delta_st == 'high' and theta_st == 'high' and alpha_st == 'low':
        findings.append({
            'condition': 'Suspected Cognitive Decline / Dementia',
            'confidence': 'MEDIUM',
            'evidence': [f'Delta elevated ({delta_pct}%)', f'Theta elevated ({theta_pct}%)', f'Alpha low ({alpha_pct}%)'],
            'recommend': 'Recommend cognitive and memory evaluation',
        })

    if beta_st == 'high' and gamma_st == 'high':
        findings.append({
            'condition': 'Suspected Anxiety / OCD',
            'confidence': 'LOW',
            'evidence': [f'Beta elevated ({beta_pct}%)', f'Gamma elevated ({gamma_pct}%)'],
            'recommend': 'Recommend mental health evaluation',
        })

    if not findings:
        findings.append({
            'condition': 'No significant abnormal pattern detected',
            'confidence': 'NORMAL',
            'evidence': ['All brainwave bands within normal range', 'No clear MRI abnormality'],
            'recommend': 'Results appear normal. Please confirm with a specialist.',
        })

    return findings
