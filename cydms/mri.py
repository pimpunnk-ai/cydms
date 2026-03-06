"""
cydms.mri
---------
MRI processing — skull stripping, mesh generation, cortical thickness
"""
import numpy as np
import nibabel as nib
import gc
from scipy import ndimage
from scipy.ndimage import label
from skimage import measure


def process_mri(mri_path):
    img = nib.load(mri_path)
    m_data = img.get_fdata().astype(np.float32)
    m_ds = m_data[::1, ::1, ::1]

    voxel_size = np.abs(img.header.get_zooms()[:3])
    min_vox = float(np.min(voxel_size))
    lo_pct = 77 if min_vox >= 0.9 else 50
    lo_thresh = np.percentile(m_ds[m_ds > 0], lo_pct)
    hi_thresh = np.percentile(m_ds[m_ds > 0], 98)

    brain_mask = (m_ds > lo_thresh) & (m_ds < hi_thresh)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = ndimage.binary_erosion(brain_mask, iterations=5)
    brain_mask = ndimage.binary_dilation(brain_mask, iterations=2)
    brain_mask = ndimage.binary_fill_holes(brain_mask)

    labeled, n = label(brain_mask)
    if n > 0:
        sizes = [np.sum(labeled == i) for i in range(1, n+1)]
        brain_mask = labeled == (np.argmax(sizes) + 1)

    m_clean = m_ds * brain_mask.astype(np.float32)
    del m_data; gc.collect()

    verts, all_faces, _, _ = measure.marching_cubes(m_clean, level=np.percentile(m_clean[m_clean > 0], 40))
    v_center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    verts_final = (verts - v_center) * 2.0
    del m_clean; gc.collect()

    # Cortical thickness
    from scipy import ndimage as ndi
    img_thick = nib.load(mri_path)
    mri_thick = img_thick.get_fdata().astype(np.float32)
    mri_ds = mri_thick[::4, ::4, ::4]

    lo_t = np.percentile(mri_ds[mri_ds > 0], 20)
    hi_t = np.percentile(mri_ds[mri_ds > 0], 98)
    mask_t = (mri_ds > lo_t) & (mri_ds < hi_t)
    mask_t = ndi.binary_fill_holes(mask_t)
    mask_t = ndi.binary_erosion(mask_t, iterations=1)

    def get_lobe_mask(arr, lobe):
        h = arr.shape
        if lobe == 'frontal':   return arr[:, int(h[1]*0.6):, :]
        if lobe == 'occipital': return arr[:, :int(h[1]*0.2), :]
        if lobe == 'temporal':  return arr[:, int(h[1]*0.2):int(h[1]*0.5), :int(h[2]*0.4)]
        return arr[:, int(h[1]*0.2):int(h[1]*0.6), int(h[2]*0.3):]

    thickness = {}
    for lobe in ['frontal', 'occipital', 'temporal', 'parietal']:
        lobe_mask = get_lobe_mask(mask_t, lobe)
        thickness[lobe] = float(np.sum(lobe_mask)) / max(lobe_mask.size, 1) * 100

    left_sum  = float(np.sum(mask_t[:mask_t.shape[0]//2, :, :]))
    right_sum = float(np.sum(mask_t[mask_t.shape[0]//2:, :, :]))
    asymmetry = abs(left_sum - right_sum) / max(left_sum + right_sum, 1)
    del mri_thick, mri_ds, mask_t; gc.collect()

    return {
        'verts': verts_final.tolist(),
        'faces': all_faces.tolist(),
        'thickness': thickness,
        'asymmetry': round(asymmetry * 100, 2),
        'voxel_size': min_vox,
        'threshold_pct': lo_pct,
    }
