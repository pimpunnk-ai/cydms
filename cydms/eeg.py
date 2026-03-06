"""
cydms.eeg
---------
EEG loading, montage setup, channel matching
"""
import numpy as np
import mne
import pandas as pd
import re
import os


def load_eeg(eeg_path, tsv_elec_path=None):
    debug_log = []
    warnings = []

    eeg_ext = os.path.splitext(eeg_path)[1].lower()
    if eeg_ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=False)
        debug_log.append("📄 EEG format: BrainVision (.vhdr)")
    elif eeg_ext == '.edf':
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
        debug_log.append("📄 EEG format: EDF (.edf)")
    elif eeg_ext == '.bdf':
        raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose=False)
        debug_log.append("📄 EEG format: BioSemi (.bdf)")
    else:
        try:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
            debug_log.append("📄 EEG format: EEGLAB (.set) — raw")
        except Exception as e_raw:
            if 'number of trials' in str(e_raw).lower() or 'epochs' in str(e_raw).lower():
                epochs = mne.io.read_epochs_eeglab(eeg_path, verbose=False)
                raw = mne.EpochsArray(epochs.get_data(), epochs.info).average().interpolate_bads()
                raw = mne.io.RawArray(np.tile(raw.data, (1, 10)), raw.info)
                debug_log.append(f"📄 EEG format: EEGLAB (.set) — epoched ({len(epochs)} trials)")
                warnings.append("⚠️ Epoched data detected — converted to continuous automatically")
            else:
                raise e_raw

    has_builtin_pos = False
    try:
        builtin_pos = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']
                      if not np.all(ch['loc'][:3] == 0) and not np.any(np.isnan(ch['loc'][:3]))}
        if len(builtin_pos) >= 3:
            has_builtin_pos = True
            debug_log.append(f"✅ พบพิกัดในไฟล์ EEG เลย: {len(builtin_pos)} channels — ไม่ต้องใช้ TSV")
    except:
        pass

    debug_log.append(f"⏱️ ใช้ข้อมูลทั้งหมด {raw.times[-1]:.1f}s")

    for ch in raw.ch_names:
        if any(x in ch.lower() for x in ['heog', 'veog']):
            raw.set_channel_types({ch: 'eog'})

    applied_tsv = False
    raw_names_before = list(raw.ch_names)

    if has_builtin_pos:
        applied_tsv = True

    if not applied_tsv and tsv_elec_path:
        try:
            def try_parse_no_separator(lines):
                rows = []
                pat = re.compile(
                    r'([A-Za-z]+\d{1,4}|[A-Za-z]{1,4}\d{0,4})'
                    r'.*?'
                    r'(-?\d{1,3}\.\d{1,4})'
                    r'[\s,]*'
                    r'(-?\d{1,3}\.\d{1,4})'
                    r'[\s,]*'
                    r'(-?\d{1,3}\.\d{1,4})',
                    re.IGNORECASE
                )
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    m = pat.search(line)
                    if m:
                        rows.append([m.group(1), m.group(2), m.group(3), m.group(4)])
                if not rows:
                    return None
                df = pd.DataFrame(rows, columns=['name', 'x', 'y', 'z'])
                return df.replace(['n/a', 'N/A', 'nan'], np.nan)

            def load_file(enc):
                with open(tsv_elec_path, 'r', encoding=enc, errors='ignore') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                parsed = [list(filter(None, re.split(r'\s+', l))) for l in lines]
                data_rows = [p for p in parsed if p and str(p[0]).upper() not in ['NAME','LABEL','N/A']]
                looks_no_sep = data_rows and all(len(p) < 3 for p in data_rows[:5])
                if looks_no_sep:
                    df_try = try_parse_no_separator(lines)
                    if df_try is not None and len(df_try) > 0:
                        return df_try
                if not parsed: return pd.DataFrame()
                df = pd.DataFrame(parsed)
                first_row = [str(x).lower() for x in parsed[0]]
                if any(c in first_row for c in ['x', 'y', 'z', 'name', 'label']):
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
                else:
                    df.columns = ['name', 'x', 'y', 'z'] + [f'extra_{i}' for i in range(len(df.columns)-4)]
                return df.replace(['n/a','N/A','nan'], np.nan)

            try:
                df_elec = load_file('utf-8-sig')
            except:
                df_elec = load_file('cp874')

            debug_log.append(f"📄 TSV: อ่านได้ {len(df_elec)} แถว | คอลัมน์: {', '.join(map(str, df_elec.columns.tolist()))}")

            cols_lower = [str(c).lower().strip() for c in df_elec.columns]
            has_header = any(re.fullmatch(r'x|y|z|name|label|pos_x|pos_y|pos_z', c) for c in cols_lower)

            if not has_header:
                df_elec = pd.read_csv(tsv_elec_path, sep=None, engine='python', header=None, na_values=['n/a','N/A','nan'])
                if len(df_elec.columns) <= 1:
                    df_elec = pd.read_csv(tsv_elec_path, sep=r'\s+', engine='python', header=None, na_values=['n/a','N/A','nan'])
                df_elec.columns = ['name', 'x', 'y', 'z'] + [f'extra_{i}' for i in range(len(df_elec.columns)-4)]

            cols = {str(c).lower().strip(): c for c in df_elec.columns}
            name_col = next((cols[c] for c in ['name', 'label', 'channel', 'ch_name', 'electrode'] if c in cols), df_elec.columns[0])
            x_col = next((cols[c] for c in ['x', 'pos_x', 'coordinate_x', 'left-right'] if c in cols), None)
            y_col = next((cols[c] for c in ['y', 'pos_y', 'coordinate_y', 'posterior-anterior'] if c in cols), None)
            z_col = next((cols[c] for c in ['z', 'pos_z', 'coordinate_z', 'inferior-superior'] if c in cols), None)

            if x_col is None and len(df_elec.columns) >= 4:
                x_col, y_col, z_col = df_elec.columns[1], df_elec.columns[2], df_elec.columns[3]

            if not all([x_col, y_col, z_col]):
                col_names = ", ".join(map(str, df_elec.columns))
                raise ValueError(f"ระบบหาตำแหน่ง x, y, z ในไฟล์ไม่พบครับ\n\n📍 ตารางที่ตรวจพบ: {col_names}")

            def normalize(name):
                s = str(name).upper().strip()
                s = re.sub(r'^(EEG|CH|ELE|REF|EOG)\s*0*', '', s)
                s = re.sub(r'[-_\s](EEG|CH|ELE|REF|EOG)$', '', s)
                return s

            def get_digits(name):
                digits = "".join(re.findall(r'\d+', str(name)))
                return str(int(digits)) if digits else ""

            final_ch_pos = {}
            rename_map = {}
            tsv_samples = []

            for _, row in df_elec.iterrows():
                tsv_name_raw = str(row[name_col]).strip()
                if tsv_name_raw.lower() in ['n/a', 'nan', '']: continue
                tsv_samples.append(tsv_name_raw)

                matched_raw = None
                for r_name in raw_names_before:
                    if r_name.upper() == tsv_name_raw.upper():
                        matched_raw = r_name; break
                if not matched_raw:
                    tsv_norm = normalize(tsv_name_raw)
                    for r_name in raw_names_before:
                        if normalize(r_name) == tsv_norm:
                            matched_raw = r_name; break
                if not matched_raw:
                    tsv_digits = get_digits(tsv_name_raw)
                    if tsv_digits:
                        for r_name in raw_names_before:
                            if get_digits(r_name) == tsv_digits:
                                matched_raw = r_name; break

                if matched_raw:
                    try:
                        px, py, pz = float(row[x_col]), float(row[y_col]), float(row[z_col])
                    except (ValueError, TypeError):
                        continue
                    pos = np.array([px, py, pz]) / 100.0
                    final_ch_pos[tsv_name_raw] = pos
                    rename_map[matched_raw] = tsv_name_raw

            if rename_map:
                raw.rename_channels(rename_map)
                raw.set_channel_types({ch: 'eeg' for ch in rename_map.values()})
                std_1020 = mne.channels.make_standard_montage('standard_1020')
                fid_pos = std_1020.get_positions()
                montage = mne.channels.make_dig_montage(
                    ch_pos=final_ch_pos,
                    nasion=fid_pos.get('nasion'), lpa=fid_pos.get('lpa'), rpa=fid_pos.get('rpa'),
                    coord_frame='head'
                )
                raw.set_montage(montage, on_missing='warn')
                applied_tsv = True
                debug_log.append(f"✅ จับคู่ชื่อสำเร็จ: {len(rename_map)} electrodes")
            else:
                valid_rows = []
                for _, row in df_elec.iterrows():
                    try:
                        px, py, pz = float(row[x_col]), float(row[y_col]), float(row[z_col])
                        valid_rows.append((px, py, pz))
                    except (ValueError, TypeError):
                        valid_rows.append(None)

                order_ch_pos = {}
                order_rename = {}
                paired = 0
                for i, r_name in enumerate(raw_names_before):
                    if i >= len(valid_rows): break
                    if valid_rows[i] is None: continue
                    px, py, pz = valid_rows[i]
                    tsv_name_order = str(df_elec.iloc[i][name_col]).strip()
                    order_ch_pos[tsv_name_order] = np.array([px, py, pz]) / 100.0
                    order_rename[r_name] = tsv_name_order
                    paired += 1

                if paired >= 3:
                    raw.rename_channels(order_rename)
                    raw.set_channel_types({ch: 'eeg' for ch in order_rename.values()})
                    std_1020 = mne.channels.make_standard_montage('standard_1020')
                    fid_pos = std_1020.get_positions()
                    montage = mne.channels.make_dig_montage(
                        ch_pos=order_ch_pos,
                        nasion=fid_pos.get('nasion'), lpa=fid_pos.get('lpa'), rpa=fid_pos.get('rpa'),
                        coord_frame='head'
                    )
                    raw.set_montage(montage, on_missing='warn')
                    applied_tsv = True
                    debug_log.append(f"⚠️ จับคู่ด้วยลำดับ: {paired} electrodes")
                    warnings.append(f"⚠️ ระบบจับคู่ขั้วไฟฟ้าด้วยลำดับ เนื่องจากชื่อไม่ตรงกัน")
                else:
                    raise ValueError(f"ไม่สามารถจับคู่ขั้วไฟฟ้าได้ครับ")

        except Exception as e_tsv:
            if isinstance(e_tsv, ValueError): raise e_tsv
            debug_log.append(f"❌ TSV error: {str(e_tsv)[:200]}")

    matched_std = []
    if not applied_tsv:
        std = mne.channels.make_standard_montage('standard_1020')
        std_names_upper = [ch.upper() for ch in std.ch_names]
        matched_std = [ch for ch in raw.ch_names if ch.upper() in std_names_upper]

        if len(matched_std) >= 3:
            raw.set_montage(std, on_missing='ignore')
            debug_log.append(f"✅ ใช้ standard_1020: match ได้ {len(matched_std)} channels")
        else:
            std_ch_names = std.ch_names
            ch_pos_std = {ch: std.get_positions()['ch_pos'][ch] for ch in std_ch_names}
            order_ch_pos = {}
            order_rename = {}
            for i, r_name in enumerate(raw.ch_names):
                if i >= len(std_ch_names): break
                std_name = std_ch_names[i]
                order_ch_pos[std_name] = ch_pos_std[std_name]
                order_rename[r_name] = std_name

            if len(order_rename) >= 3:
                raw.rename_channels(order_rename)
                raw.set_montage(std, on_missing='ignore')
                applied_tsv = True
                warnings.append(f"⚠️ ไม่มีไฟล์ TSV — ระบบจับคู่ด้วยลำดับกับ standard_1020")
                debug_log.append(f"⚠️ order-based กับ standard_1020: {len(order_rename)} channels")
            else:
                raise ValueError(f"ชื่อขั้วไฟฟ้าในไฟล์ไม่ตรงกับมาตรฐาน 10-20 และไม่มีไฟล์พิกัดที่ใช้งานได้ครับ")

    raw.filter(1, 45, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')

    valid_chs = [ch['ch_name'] for ch in raw.info['chs']
                 if not np.all(ch['loc'][:3] == 0) and not np.any(np.isnan(ch['loc'][:3]))]

    if len(valid_chs) < 3:
        raise ValueError(f"พบขั้วไฟฟ้าที่มีพิกัดน้อยเกินไป (พบแค่ {len(valid_chs)} จุด)")

    raw.pick_channels(valid_chs)
    raw.set_eeg_reference('average', projection=True).apply_proj()

    return raw, debug_log, warnings
