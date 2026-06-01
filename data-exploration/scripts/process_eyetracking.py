"""
Process raw eye-tracking CSV into fixations, saccades, and metrics CSVs.

Usage:
    python process_eyetracking.py data/Subject_1003_T4_Meaningful_Text_raw.csv \
        --roi data/Meaningful_Text_rois.csv
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import I2MC

# ── I2MC / recording parameters ───────────────────────────────────────────────
FREQ_HZ      = 250
MIN_FIX_MS   = 40
SCREEN_X     = 1920
SCREEN_Y     = 1080
MISSING_VAL  = -32768   # sentinel used to mark missing samples for I2MC

# Pixels per visual-angle degree for velocity conversion.
# Derived empirically from the reference saccade file (~37.5 px/deg).
# Set to None to output velocities in px/ms instead.
PIX_PER_DEG  = 37.5


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_filename(raw_path):
    """Return (sid, task, stimfile) from a raw CSV path."""
    base = os.path.basename(raw_path)               # Subject_1003_T4_Meaningful_Text_raw.csv
    m = re.match(r'Subject_(\d+)_(.+)_raw\.csv', base)
    if not m:
        sys.exit(f"Filename {base!r} does not match 'Subject_<sid>_<task>_raw.csv'")
    sid  = int(m.group(1))
    task = m.group(2)
    # stimfile convention seen in reference data
    task_lower = task.lower().replace('_', '')
    m2 = re.match(r't(\d+)', task_lower)
    t_num = m2.group(1) if m2 else '1'
    stimfile = f"s7_stimuli_t{t_num}.jpg"
    return sid, task, stimfile


def normalize_roi_name(name: str) -> str:
    """'line 1 part 3' -> 'line_001-part_003' ;  'line 2' -> 'line_002'"""
    name = name.strip()
    m = re.match(r'line\s+(\d+)\s+part\s+(\d+)', name, re.I)
    if m:
        return f"line_{int(m.group(1)):03d}-part_{int(m.group(2)):03d}"
    m = re.match(r'line\s+(\d+)', name, re.I)
    if m:
        return f"line_{int(m.group(1)):03d}"
    return name


def load_rois(roi_path, stimfile):
    """Load ROI CSV and return (all_rois df, line_rois df with center_y and y_thresh)."""
    df = pd.read_csv(roi_path)
    df = df[df['stimfile'] == stimfile].copy().reset_index(drop=True)
    df['aoi_name'] = df['name'].apply(normalize_roi_name)
    lines = df[df['kind'] == 'line'].copy().reset_index(drop=True)
    lines['center_y'] = lines['y'] + lines['height'] / 2.0
    # Threshold for assigning a fixation to a line = midpoint between adjacent
    # line centres.  For the first/last line we extend outward by half-spacing.
    centres = lines['center_y'].sort_values().values
    spacings = np.diff(centres)
    half_sp  = np.median(spacings) / 2.0   # typically ~46 px for this stimulus
    lines = lines.sort_values('center_y').reset_index(drop=True)
    lines['y_lo'] = lines['center_y'] - half_sp
    lines['y_hi'] = lines['center_y'] + half_sp
    return df, lines


def assign_aoi(orig_x, orig_y, rois_df, lines_df):
    """
    Returns (aoi_subline, aoi_line, fix_y_snapped).

    fix_y is snapped to the center-y of the nearest line regardless of
    whether orig_y falls within that line's bounding box.
    AOI labels are assigned only when (orig_x, orig_y) is inside the ROI bbox.
    """
    # Nearest line (by y-distance to centre)
    dists = (lines_df['center_y'] - orig_y).abs()
    nearest = lines_df.loc[dists.idxmin()]
    fix_y   = float(nearest['center_y'])

    aoi_subline = ''
    aoi_line    = ''

    # Assign aoi_line using the midpoint-between-lines threshold so that
    # fixations in the inter-line gaps still get a line assignment.
    for _, lroi in lines_df.iterrows():
        if lroi['y_lo'] <= orig_y <= lroi['y_hi']:
            # Also require x within the line's x span
            if lroi['x'] <= orig_x <= lroi['x'] + lroi['width']:
                aoi_line = lroi['aoi_name']
                break

    # Assign aoi_subline using strict bounding box
    if aoi_line:
        sublines = rois_df[rois_df['kind'] == 'sub-line']
        for _, roi in sublines.iterrows():
            if (roi['x'] <= orig_x <= roi['x'] + roi['width'] and
                    roi['y'] <= orig_y <= roi['y'] + roi['height']):
                aoi_subline = roi['aoi_name']
                break

    return aoi_subline, aoi_line, fix_y


# ── step 1: run I2MC ───────────────────────────────────────────────────────────

def run_i2mc(raw_df):
    # Two-eye mode (L + R separately) gives the closest fixation count to the
    # reference (~139-140 vs reference 141).  Merge parameters match I2MC defaults.
    gaze = pd.DataFrame({
        'time': raw_df['time'] / 1000.0,   # µs → ms
        'L_X' : raw_df['gaze_x_left'],
        'L_Y' : raw_df['gaze_y_left'],
        'R_X' : raw_df['gaze_x_right'],
        'R_Y' : raw_df['gaze_y_right'],
    })
    options = {
        'xres'        : SCREEN_X,
        'yres'        : SCREEN_Y,
        'freq'        : FREQ_HZ,
        'missingx'    : MISSING_VAL,
        'missingy'    : MISSING_VAL,
        'minFixDur'   : MIN_FIX_MS,
        'maxMergeDist': 30,    # px — default; adjust to tune fixation count
        'maxMergeTime': 30,    # ms — default
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = I2MC.I2MC(gaze, options, logging=False)

    if result is False or result[0] is False:
        sys.exit("I2MC clustering failed.")
    fix, _, _ = result
    return fix


# ── step 2: fixations CSV ──────────────────────────────────────────────────────

def build_fixations(fix, rois_df, lines_df, task, sid, stimfile, trialid):
    n    = len(fix['startT'])
    rows = []
    for i in range(n):
        ox = float(fix['xpos'][i])
        oy = float(fix['ypos'][i])
        aoi_sub, aoi_line, fix_y = assign_aoi(ox, oy, rois_df, lines_df)
        rows.append({
            'id'         : i,
            'task'       : task,
            'sid'        : sid,
            'eye'        : 'b',
            'stimfile'   : stimfile,
            'trialid'    : trialid,
            'start_ms'   : fix['startT'][i],
            'end_ms'     : fix['endT'][i],
            'duration_ms': fix['dur'][i],
            'fix_x'      : ox,
            'fix_y'      : fix_y,
            'orig_fix_x' : ox,
            'orig_fix_y' : oy,
            'disp_x'     : fix.get('fixRangeX', [np.nan]*n)[i],
            'disp_y'     : fix.get('fixRangeY', [np.nan]*n)[i],
            'aoi_subline': aoi_sub,
            'aoi_line'   : aoi_line,
        })
    return pd.DataFrame(rows)


# ── step 3: saccades CSV ───────────────────────────────────────────────────────

def _binocular_avg(raw_df):
    """Return arrays of (time_ms, avg_x, avg_y) from raw binocular gaze."""
    t = raw_df['time'].values / 1000.0
    x = (raw_df['gaze_x_left'].values + raw_df['gaze_x_right'].values) / 2.0
    y = (raw_df['gaze_y_left'].values + raw_df['gaze_y_right'].values) / 2.0
    return t, x, y


def _saccade_velocity(t_seg, x_seg, y_seg):
    """
    Compute per-sample velocities for a saccade segment.

    Returns (avg_vel, avg_vel_x, avg_vel_y, peak_vel, peak_vel_x, peak_vel_y).
    Velocities in deg/s when PIX_PER_DEG is set, otherwise in px/ms.
    """
    if len(t_seg) < 2:
        return (0.0,) * 6

    dt = np.diff(t_seg)                 # ms
    dx = np.diff(x_seg)
    dy = np.diff(y_seg)

    # avoid division by zero
    dt = np.where(dt == 0, 1e-6, dt)

    vx = np.abs(dx) / dt               # px/ms
    vy = np.abs(dy) / dt
    vm = np.sqrt(dx**2 + dy**2) / dt   # magnitude

    if PIX_PER_DEG is not None:
        scale = 1000.0 / PIX_PER_DEG   # px/ms → deg/s
        vx *= scale; vy *= scale; vm *= scale

    peak_idx = int(np.argmax(vm))
    return (
        float(np.mean(vm)),
        float(np.mean(vx)),
        float(np.mean(vy)),
        float(vm[peak_idx]),
        float(vx[peak_idx]),
        float(vy[peak_idx]),
    )


def build_saccades(fix_df, raw_df, task, sid, stimfile, trialid):
    t_raw, x_raw, y_raw = _binocular_avg(raw_df)
    t_raw_us = raw_df['time'].values     # microseconds for binary search

    rows = []
    n_fix = len(fix_df)

    for i in range(n_fix - 1):
        f0 = fix_df.iloc[i]
        f1 = fix_df.iloc[i + 1]

        sacc_start_ms = f0['end_ms']
        sacc_end_ms   = f1['start_ms']
        duration_ms   = sacc_end_ms - sacc_start_ms

        # Raw samples that fall strictly within the saccade interval
        mask = (t_raw >= sacc_start_ms) & (t_raw <= sacc_end_ms)
        t_seg = t_raw[mask]
        x_seg = x_raw[mask]
        y_seg = y_raw[mask]

        # Boundary positions: first / last sample of saccade segment
        if len(x_seg) >= 1:
            sx, sy = float(x_seg[0]),  float(y_seg[0])
            ex, ey = float(x_seg[-1]), float(y_seg[-1])
        else:
            # No raw samples in gap → use fixation medians
            sx, sy = float(f0['orig_fix_x']), float(f0['orig_fix_y'])
            ex, ey = float(f1['orig_fix_x']), float(f1['orig_fix_y'])

        ampl_x = ex - sx
        ampl_y = ey - sy
        ampl   = float(np.sqrt(ampl_x**2 + ampl_y**2))

        avg_vel, avg_vel_x, avg_vel_y, peak_vel, peak_vel_x, peak_vel_y = \
            _saccade_velocity(t_seg, x_seg, y_seg)

        # Column order matches the reference file exactly:
        # start_x, end_y, start_y, ampl_x, ampl_y, ..., end_x
        rows.append({
            'id'         : i,
            'task'       : task,
            'sid'        : sid,
            'eye'        : 'b',
            'stimfile'   : stimfile,
            'trialid'    : trialid,
            'start_ms'   : sacc_start_ms,
            'end_ms'     : sacc_end_ms,
            'duration_ms': duration_ms,
            'start_x'    : sx,
            'end_y'      : ey,       # note: column order mirrors reference (end_y before start_y)
            'start_y'    : sy,
            'ampl_x'     : ampl_x,
            'ampl_y'     : ampl_y,
            'avg_vel_x'  : avg_vel_x,
            'avg_vel_y'  : avg_vel_y,
            'peak_vel_x' : peak_vel_x,
            'peak_vel_y' : peak_vel_y,
            'end_x'      : ex,
            'avg_vel'    : avg_vel,
            'peak_vel'   : peak_vel,
            'ampl'       : ampl,
        })

    return pd.DataFrame(rows)


# ── step 4: metrics CSV ────────────────────────────────────────────────────────

def _classify_saccades(fix_df, sacc_df):
    """
    Return (n_progress, n_regress_within, n_regress_between, n_transit).

    Progressive  : within same line, moving right (ampl_x > 0)
    Within-regress: within same line, moving left  (ampl_x < 0)
    Between-regress: crossing to an earlier line (lower line number)
    Transit      : progressive + within-regress (all within-line saccades)
    """
    n_progress = n_within_reg = n_between_reg = 0

    for i in range(len(sacc_df)):
        if i + 1 >= len(fix_df):
            break
        line0 = fix_df.iloc[i]['aoi_line']
        line1 = fix_df.iloc[i + 1]['aoi_line']

        if pd.isna(line0) or pd.isna(line1) or line0 == '' or line1 == '':
            continue   # fixation outside text area

        if line0 == line1:
            # within-line saccade
            if sacc_df.iloc[i]['ampl_x'] > 0:
                n_progress += 1
            else:
                n_within_reg += 1
        else:
            # between-line — check if it's a regression (going back)
            def line_num(s):
                m = re.search(r'(\d+)', s)
                return int(m.group(1)) if m else 0
            if line_num(line1) < line_num(line0):
                n_between_reg += 1

    n_transit = n_progress + n_within_reg
    return n_progress, n_within_reg, n_between_reg, n_transit


def _visit_groups(fix_indices):
    """
    Given a sorted list of fixation indices belonging to one AOI,
    split into contiguous visit groups (broken by fixations outside the AOI).
    Returns list of lists of indices.
    """
    if not fix_indices:
        return []
    groups = []
    current = [fix_indices[0]]
    for idx in fix_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return groups


def build_metrics(fix_df, sacc_df, rois_df, task, sid, stimfile, trialid):
    n_fix   = len(fix_df)
    n_sacc  = len(sacc_df)

    # ── whole-trial metrics ────────────────────────────────────────────────────
    sum_fix_dur_trial  = float(fix_df['duration_ms'].sum())
    dwell_time_trial   = (float(fix_df.iloc[-1]['end_ms']) -
                          float(fix_df.iloc[0]['start_ms'])) if n_fix > 0 else 0.0
    mean_fix_dur_trial = sum_fix_dur_trial / n_fix if n_fix > 0 else 0.0

    sum_sacc_dur_trial  = float(sacc_df['duration_ms'].sum()) if n_sacc > 0 else 0.0
    mean_sacc_dur_trial = sum_sacc_dur_trial / n_sacc if n_sacc > 0 else 0.0
    mean_sacc_ampl_trial= float(sacc_df['ampl'].mean()) if n_sacc > 0 else 0.0

    n_progress, n_within_reg, n_between_reg, n_transit = \
        _classify_saccades(fix_df, sacc_df)
    n_regress = n_within_reg + n_between_reg
    ratio_pr  = (float(n_progress) / n_within_reg
                 if n_within_reg > 0 else float('inf'))

    trial_cols = {
        'sid'                        : sid,
        'stimfile'                   : stimfile,
        'eye_used'                   : 'b',
        'trialid'                    : trialid,
        'n_fix_trial'                : n_fix,
        'sum_fix_dur_trial'          : sum_fix_dur_trial,
        'dwell_time_trial'           : dwell_time_trial,
        'mean_fix_dur_trial'         : mean_fix_dur_trial,
        'n_sacc_trial'               : n_sacc,
        'sum_sacc_dur_trial'         : sum_sacc_dur_trial,
        'mean_sacc_dur_trial'        : mean_sacc_dur_trial,
        'mean_sacc_ampl_trial'       : mean_sacc_ampl_trial,
        'ratio_progress_regress_trial': ratio_pr,
        'n_between_line_regress_trial': n_between_reg,
        'n_within_line_regress_trial' : n_within_reg,
        'n_regress_trial'            : n_regress,
        'n_progress_trial'           : n_progress,
        'n_transit_trial'            : n_transit,
    }

    # ── per-ROI metrics ────────────────────────────────────────────────────────
    rows = []

    # Iterate over every ROI (lines first, then sublines per reference ordering)
    for _, roi in rois_df.iterrows():
        aoi_name = roi['aoi_name']
        content  = roi.get('content', '')
        kind_map = {'line': 'line', 'sub-line': 'subline'}
        aoi_kind = kind_map.get(roi['kind'], roi['kind'])

        # Fixations belonging to this ROI
        if aoi_kind == 'line':
            fix_mask = fix_df['aoi_line'] == aoi_name
        else:
            fix_mask = fix_df['aoi_subline'] == aoi_name

        aoi_fix_idx  = list(fix_df.index[fix_mask])
        aoi_fix      = fix_df.loc[aoi_fix_idx]

        n_fix_aoi = len(aoi_fix)
        skipped   = 1 if n_fix_aoi == 0 else 0

        if n_fix_aoi == 0:
            row = {**trial_cols,
                   'aoi': aoi_name, 'aoi_kind': aoi_kind, 'content': content,
                   'dwell_time_aoi': 0.0, 'n_fix_aoi': 0,
                   'sum_fix_dur_aoi': 0.0, 'mean_fix_dur_aoi': 0.0,
                   'skipped_aoi': skipped,
                   'n_fix_first_visit_aoi': 0, 'first_fix_dur_aoi': 0.0,
                   'first_fix_land_pos_aoi': np.nan,
                   'dwell_time_first_visit_aoi': 0.0,
                   'sum_fix_dur_first_visit_aoi': 0.0,
                   'sum_fix_dur_after_first_visit_aoi': 0.0,
                   'dwell_time_rereading_aoi': 0.0,
                   'n_revisits_aoi': 0,
                   'task': task}
            rows.append(row)
            continue

        # Visit groups (contiguous fixation runs within this AOI)
        visits = _visit_groups(aoi_fix_idx)
        n_revisits = len(visits) - 1

        # Dwell time = sum of (last_end - first_start) per visit
        dwell_time_aoi = sum(
            fix_df.loc[v[-1], 'end_ms'] - fix_df.loc[v[0], 'start_ms']
            for v in visits
        )

        # sum_fix_dur_aoi is 2× actual sum (matches reference behaviour)
        actual_sum_dur = float(aoi_fix['duration_ms'].sum())
        sum_fix_dur_aoi  = 2.0 * actual_sum_dur
        mean_fix_dur_aoi = sum_fix_dur_aoi / n_fix_aoi

        # First visit stats
        first_visit_idx = visits[0]
        first_visit_fix = fix_df.loc[first_visit_idx]

        n_fix_first_visit   = len(first_visit_idx)
        first_fix           = fix_df.loc[first_visit_idx[0]]
        first_fix_dur       = float(first_fix['duration_ms'])
        dwell_time_first    = (fix_df.loc[first_visit_idx[-1], 'end_ms'] -
                               fix_df.loc[first_visit_idx[0],  'start_ms'])
        sum_fix_dur_first   = float(first_visit_fix['duration_ms'].sum())

        # Landing position of first fixation within the ROI
        if aoi_kind == 'subline' and roi['width'] > 0:
            land_pos = (float(first_fix['fix_x']) - float(roi['x'])) / float(roi['width'])
        elif aoi_kind == 'line' and roi['width'] > 0:
            land_pos = (float(first_fix['fix_x']) - float(roi['x'])) / float(roi['width'])
        else:
            land_pos = np.nan

        # Revisit stats (all visits after the first)
        if n_revisits > 0:
            revisit_idx = [idx for v in visits[1:] for idx in v]
            revisit_fix = fix_df.loc[revisit_idx]
            sum_fix_dur_after = float(revisit_fix['duration_ms'].sum())
            dwell_rereading   = sum(
                fix_df.loc[v[-1], 'end_ms'] - fix_df.loc[v[0], 'start_ms']
                for v in visits[1:]
            )
        else:
            sum_fix_dur_after = 0.0
            dwell_rereading   = 0.0

        row = {**trial_cols,
               'aoi': aoi_name, 'aoi_kind': aoi_kind, 'content': content,
               'dwell_time_aoi'                  : dwell_time_aoi,
               'n_fix_aoi'                       : n_fix_aoi,
               'sum_fix_dur_aoi'                 : sum_fix_dur_aoi,
               'mean_fix_dur_aoi'                : mean_fix_dur_aoi,
               'skipped_aoi'                     : skipped,
               'n_fix_first_visit_aoi'           : n_fix_first_visit,
               'first_fix_dur_aoi'               : first_fix_dur,
               'first_fix_land_pos_aoi'          : land_pos,
               'dwell_time_first_visit_aoi'      : dwell_time_first,
               'sum_fix_dur_first_visit_aoi'     : sum_fix_dur_first,
               'sum_fix_dur_after_first_visit_aoi': sum_fix_dur_after,
               'dwell_time_rereading_aoi'        : dwell_rereading,
               'n_revisits_aoi'                  : n_revisits,
               'task'                            : task}
        rows.append(row)

    col_order = [
        'sid', 'stimfile', 'eye_used', 'trialid',
        'n_fix_trial', 'sum_fix_dur_trial', 'dwell_time_trial', 'mean_fix_dur_trial',
        'n_sacc_trial', 'sum_sacc_dur_trial', 'mean_sacc_dur_trial', 'mean_sacc_ampl_trial',
        'ratio_progress_regress_trial',
        'n_between_line_regress_trial', 'n_within_line_regress_trial',
        'n_regress_trial', 'n_progress_trial', 'n_transit_trial',
        'aoi', 'aoi_kind', 'content',
        'dwell_time_aoi', 'n_fix_aoi', 'sum_fix_dur_aoi', 'mean_fix_dur_aoi',
        'skipped_aoi',
        'n_fix_first_visit_aoi', 'first_fix_dur_aoi', 'first_fix_land_pos_aoi',
        'dwell_time_first_visit_aoi', 'sum_fix_dur_first_visit_aoi',
        'sum_fix_dur_after_first_visit_aoi', 'dwell_time_rereading_aoi',
        'n_revisits_aoi', 'task',
    ]
    return pd.DataFrame(rows)[col_order]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('raw_csv',  help='Path to Subject_*_raw.csv')
    ap.add_argument('--roi',    required=True, help='Path to *_rois.csv')
    ap.add_argument('--trial-id', type=int, default=1,
                    help='Trial ID to embed in output files (default: 1)')
    args = ap.parse_args()

    sid, task, stimfile = parse_filename(args.raw_csv)
    trialid = args.trial_id
    out_dir = os.path.dirname(os.path.abspath(args.raw_csv))
    prefix  = f"Subject_{sid}_{task}"

    print(f"Subject {sid} | Task {task} | Trial {trialid}")

    # 1. Load data
    print("Loading raw data …")
    raw_df = pd.read_csv(args.raw_csv)

    print("Loading ROIs …")
    rois_df, lines_df = load_rois(args.roi, stimfile)

    # 2. I2MC
    print("Running I2MC fixation detection …")
    fix = run_i2mc(raw_df)
    print(f"  → {len(fix['startT'])} fixations detected")

    # 3. Fixations
    print("Building fixations …")
    fix_df = build_fixations(fix, rois_df, lines_df, task, sid, stimfile, trialid)
    fix_path = os.path.join(out_dir, f"{prefix}_fixations.csv")
    fix_df.to_csv(fix_path, index=False)
    print(f"  → {fix_path}")

    # 4. Saccades
    print("Building saccades …")
    sacc_df = build_saccades(fix_df, raw_df, task, sid, stimfile, trialid)
    sacc_path = os.path.join(out_dir, f"{prefix}_saccades.csv")
    sacc_df.to_csv(sacc_path, index=False)
    print(f"  → {sacc_path}")

    # 5. Metrics
    print("Building metrics …")
    met_df = build_metrics(fix_df, sacc_df, rois_df, task, sid, stimfile, trialid)
    met_path = os.path.join(out_dir, f"{prefix}_metrics.csv")
    met_df.to_csv(met_path, index=False)
    print(f"  → {met_path}")

    print("Done.")


if __name__ == '__main__':
    main()
