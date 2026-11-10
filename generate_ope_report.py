#!/usr/bin/env python3
"""
generate_ope_report.py
======================
Generate an observation plan report (PDF) with plots and observation table
from an input PFS OPE file and pfsDesign files.

Outputs:
  - Page 1: Elevation vs Time (HST) & Rotator Angle vs Time (HST)
  - Page 2: Formatted observation table (Time, LST, Target, Air, Teff, El, Rot, Moon)
"""

import argparse
import datetime
import os
import re
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_body
from astropy.io import fits
from astropy.time import Time, TimeDelta
import astropy.units as u
from astroplan import Observer, moon_illumination
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image

from ope_utils import load_config, MoonBrightnessModel, calculate_teff

warnings.filterwarnings('ignore')


def parse_ope_file(ope_path: Path):
    """
    Parse the OPE file to extract observation date, parameters, and observation commands.
    """
    text = ope_path.read_text(encoding='utf-8', errors='ignore')
    
    # 1. Date extraction from Header
    date_str = None
    m_date = re.search(r"OBSERVATION_START_DATE\s*=\s*([0-9.]+)", text)
    if m_date:
        date_str = m_date.group(1).replace(".", "-")
    else:
        # Fallback to filename (e.g. 2026-11-09_ssp_v2.ope)
        m_file = re.search(r"(\d{4}-\d{2}-\d{2})", ope_path.name)
        if m_file:
            date_str = m_file.group(1)
        else:
            date_str = datetime.date.today().strftime("%Y-%m-%d")

    # 2. Parameters (RA, Dec defined for objects)
    params = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"(\w+)=OBJECT=\"([^\"]+)\"\s+RA=([0-9.]+)\s+DEC=([+-][0-9.]+)", line)
        if m:
            name, obj, ra_str, dec_str = m.groups()
            params[name] = {"ra_str": ra_str, "dec_str": dec_str}

    # 3. Observation commands
    observations = []
    current = {}
    for line in text.splitlines():
        line = line.strip()
        
        # Priority and PA from comment
        m_pa = re.match(r"###\s+(\w+)\s+PA=([0-9.]+)(?:\s+PRIORITY=(\d+))?\s+###", line)
        if m_pa:
            current["comment_target"] = m_pa.group(1)
            current["comment_pa"] = float(m_pa.group(2))
            if m_pa.group(3):
                current["priority"] = int(m_pa.group(3))
                
        # Observation start time
        m_time = re.match(r"###\s+OBSTIME:\s+([0-9T:-]+)\s+###", line)
        if m_time:
            current["obstime"] = m_time.group(1)
            
        # SETUPFIELD command
        m_setup = re.match(r"SETUPFIELD.*DESIGN_ID=\"([^\"]+)\"\s+OBJECT=\"([^\"]+)\"", line)
        if m_setup:
            current["design_id"] = m_setup.group(1)
            current["object"] = m_setup.group(2)
            
        # GETOBJECT command
        m_get = re.match(r"GETOBJECT.*EXPTIME=([0-9.]+)\s+NFRAME=(\d+)\s+SEQ_NAME=\"([^\"]+)\"", line)
        if m_get:
            current["exptime"] = float(m_get.group(1))
            current["nframe"] = int(m_get.group(2))
            current["seq_name"] = m_get.group(3)
            current["target"] = current.get("object", current.get("comment_target", current["seq_name"]))
            observations.append(current)
            current = {}
            
    return date_str, params, observations


def build_design_index(pfs_designs_dir: Path):
    """
    Index all pfsDesign-*.fits files under pfs_designs_dir.
    Maps lowercase 16-hex / int strings to file paths.
    """
    design_map = {}
    if not pfs_designs_dir.exists():
        print(f"Warning: pfsDesign directory not found: {pfs_designs_dir}")
        return design_map
        
    for p in pfs_designs_dir.rglob("pfsDesign-*.fits"):
        # e.g. pfsDesign-0x21a37ba30be48408.fits
        stem = p.stem.replace("pfsDesign-", "").lower()
        design_map[stem] = p
        try:
            val_int = int(stem, 16)
            design_map[str(val_int)] = p
            design_map[f"0x{val_int:016x}"] = p
        except ValueError:
            pass
            
    return design_map


def get_target_details(target_obs, params, design_map):
    """
    Retrieve RA, Dec (in degrees) and PA for a target using pfsDesign FITS or OPE params.
    """
    design_id = target_obs.get("design_id", "").lower()
    fits_path = design_map.get(design_id)
    if fits_path is None and design_id.startswith("0x"):
        fits_path = design_map.get(design_id[2:])
        
    ra_deg = None
    dec_deg = None
    pa_deg = target_obs.get("comment_pa", 90.0)
    
    if fits_path and fits_path.exists():
        try:
            hdr = fits.getheader(fits_path, 0)
            ra_deg = float(hdr.get("RA"))
            dec_deg = float(hdr.get("DEC"))
            if "POSANG" in hdr:
                pa_deg = float(hdr.get("POSANG"))
        except Exception as e:
            print(f"Warning: Failed to read FITS header from {fits_path}: {e}")
            
    # Fallback to OPE parameter definitions if not found in FITS
    if ra_deg is None or dec_deg is None:
        target_name = target_obs.get("target")
        if target_name in params:
            ra_str = params[target_name]["ra_str"]  # e.g. "231658.559"
            dec_str = params[target_name]["dec_str"]  # e.g. "+020411.75"
            try:
                # Parse hhmmss.sss +ddmmss.ss
                ra_h = float(ra_str[:2])
                ra_m = float(ra_str[2:4])
                ra_s = float(ra_str[4:])
                ra_deg = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0
                
                sign = -1.0 if dec_str.startswith("-") else 1.0
                dec_clean = dec_str.lstrip("+-")
                dec_d = float(dec_clean[:2])
                dec_m = float(dec_clean[2:4])
                dec_s = float(dec_clean[4:])
                dec_deg = sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
            except Exception as e:
                print(f"Warning: Failed to parse coordinate strings ({ra_str}, {dec_str}): {e}")

    return ra_deg, dec_deg, pa_deg


def calculate_schedule_parameters(observations, params, design_map, observer, mbm, slot_duration_min=20):
    """
    Compute observation parameters (LST, Airmass, Teff, Elevation, Rotator, Moon) for each observation.
    """
    rows = []
    for obs in observations:
        obstime_str = obs.get("obstime")
        target_name = obs.get("target")
        if not obstime_str or "XX" in obstime_str:
            print(f"  Note: Skipping {target_name}: OBSTIME is not specified ({obstime_str}).")
            continue
            
        # OPE OBSTIME is in HST (local Hawaii time)
        # Convert HST to UTC: UTC = HST + 10 hours
        t_start_hst = Time(obstime_str)
        t_start = t_start_hst + TimeDelta(10 * 3600, format="sec")
        slot_sec = slot_duration_min * 60
        t_end = t_start + TimeDelta(slot_sec, format="sec")
        t_mid = t_start + TimeDelta(slot_sec / 2.0, format="sec")
        
        ra_deg, dec_deg, pa_deg = get_target_details(obs, params, design_map)
        if ra_deg is None or dec_deg is None:
            print(f"Skipping {obs.get('target')}: Coordinates not found.")
            continue
            
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        
        # 1. Time (HST) string: HH:MM - HH:MM
        t_hst_start_dt = (t_start - TimeDelta(10 * 3600, format="sec")).datetime
        t_hst_end_dt = (t_end - TimeDelta(10 * 3600, format="sec")).datetime
        time_hst_str = f"{t_hst_start_dt.strftime('%H:%M')} - {t_hst_end_dt.strftime('%H:%M')}"
        
        # 2. LST at start
        lst = observer.local_sidereal_time(t_start)
        lst_str = lst.to_string(sep=":", precision=0)
        
        # 3. Airmass at mid
        altaz_mid = observer.altaz(t_mid, coord)
        altitude_mid = altaz_mid.alt.deg
        airmass = altaz_mid.secz.value if hasattr(altaz_mid.secz, "value") else altaz_mid.secz
        
        # 4. Elevation at start & end
        el_start = observer.altaz(t_start, coord).alt.deg
        el_end = observer.altaz(t_end, coord).alt.deg
        
        # 5. Rotator at start & end
        pa_start = observer.parallactic_angle(t_start, coord).deg
        pa_end = observer.parallactic_angle(t_end, coord).deg
        rot_start = (pa_start + pa_deg + 180.0) % 360.0 - 180.0
        rot_end = (pa_end + pa_deg + 180.0) % 360.0 - 180.0
        
        # 6. Moon parameters at mid
        moon_coord = get_body("moon", t_mid, location=observer.location)
        moon_sep = coord.separation(moon_coord).deg
        moon_illum = moon_illumination(t_mid)
        moon_altaz = observer.altaz(t_mid, moon_coord)
        moon_alt = moon_altaz.alt.deg
        
        # 7. Teff calculation
        sun_coord = get_body("sun", t_mid, location=observer.location)
        elongation = sun_coord.separation(moon_coord)
        moon_phase_deg = 180.0 - elongation.deg
        teff_val = calculate_teff(observer, coord, altitude_mid, airmass, moon_coord, moon_altaz, moon_phase_deg, mbm)
        
        rows.append({
            "target": obs.get("target"),
            "time_hst_str": time_hst_str,
            "t_start_hst": t_hst_start_dt,
            "t_end_hst": t_hst_end_dt,
            "t_start_utc": t_start,
            "t_end_utc": t_end,
            "coord": coord,
            "pa_deg": pa_deg,
            "lst_str": lst_str,
            "airmass": airmass,
            "teff": teff_val,
            "el_start": el_start,
            "el_end": el_end,
            "rot_start": rot_start,
            "rot_end": rot_end,
            "moon_sep": moon_sep,
            "moon_illum": moon_illum,
            "moon_alt": moon_alt,
        })
        
    return rows


def generate_plots(calculated_rows, date_str, observer, config, output_alt_png, output_rot_png):
    """
    Generate Elevation vs Time and Rotator Angle vs Time plots for the night.
    """
    twilight_horizon = config['twilight']['horizon_deg']
    min_alt = config['constraints']['min_altitude']
    max_alt = config['constraints']['max_altitude']
    rot_min = config['constraints']['rotator_min']
    rot_max = config['constraints']['rotator_max']
    
    # Night bounds: 18:00 HST to 06:00 HST next day
    anchor_date = datetime.date.fromisoformat(date_str)
    xlim_start = datetime.datetime.combine(anchor_date, datetime.time(18, 0))
    xlim_end = xlim_start + datetime.timedelta(hours=12)
    
    # Twilight computation
    total_minutes = int((xlim_end - xlim_start).total_seconds() / 60)
    minutes_grid = np.linspace(0, total_minutes, 240)
    times_hst = [xlim_start + datetime.timedelta(minutes=m) for m in minutes_grid]
    times_utc = [t + datetime.timedelta(hours=10) for t in times_hst]
    times_astropy = Time(times_utc)
    
    sun_alt = observer.altaz(times_astropy, get_body("sun", times_astropy, location=observer.location)).alt.deg
    moon_alt = observer.altaz(times_astropy, get_body("moon", times_astropy, location=observer.location)).alt.deg
    
    # Shading for twilight
    img_data = np.zeros((1, len(minutes_grid), 4))
    img_data[:, :, 0] = 1.0
    img_data[:, :, 1] = 0.7
    img_data[:, :, 2] = 0.2
    max_alpha = 0.4
    alphas = np.zeros_like(sun_alt)
    alphas[sun_alt >= 0] = max_alpha
    mask = (sun_alt < 0) & (sun_alt > twilight_horizon)
    alphas[mask] = max_alpha * (sun_alt[mask] - twilight_horizon) / (-twilight_horizon)
    img_data[:, :, 3] = alphas

    colors_list = plt.cm.tab10(np.linspace(0, 1, max(10, len(calculated_rows))))
    
    # -------------------------------------------------------------
    # 1. Altitude vs Time Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(f"Elevation vs Time - {date_str} (HST)", fontsize=13, fontweight='bold', pad=10)
    
    # Twilight background
    ax.imshow(img_data, extent=[mdates.date2num(xlim_start), mdates.date2num(xlim_end), 0, 90],
              aspect='auto', origin='lower', zorder=0)
              
    # Moon altitude curve
    ax.plot(times_hst, moon_alt, color='black', linestyle='--', alpha=0.7, label='Moon', zorder=4)
    
    # Avoidance & warning zones
    ax.fill_between([xlim_start, xlim_end], 0, min_alt, color='red', alpha=0.12, zorder=1)
    ax.axhline(min_alt, color='red', linestyle=':', alpha=0.5, label=f'Min Alt ({min_alt}°)')
    if max_alt < 90:
        ax.fill_between([xlim_start, xlim_end], max_alt, 90, color='red', alpha=0.12, zorder=1)
        ax.axhline(max_alt, color='red', linestyle=':', alpha=0.5, label=f'Max Alt ({max_alt}°)')
    
    # Target altitude trajectories & observation slots
    for i, row in enumerate(calculated_rows):
        c = colors_list[i % len(colors_list)]
        coord = row["coord"]
        tgt_alts = observer.altaz(times_astropy, coord).alt.deg
        ax.plot(times_hst, tgt_alts, color=c, alpha=0.35, linewidth=1.2, zorder=3)
        
        # Highlight active slot
        s_hst = row["t_start_hst"]
        e_hst = row["t_end_hst"]
        ax.plot([s_hst, e_hst], [row["el_start"], row["el_end"]], color=c, linewidth=3.5, zorder=6)
        ax.scatter([s_hst, e_hst], [row["el_start"], row["el_end"]], color=c, s=35, zorder=7)
        # Label target near start with alternating vertical offset and subtle white bbox
        offset_y = 7 if (i % 2 == 0) else -13
        ax.annotate(row["target"], (s_hst, row["el_start"]),
                    textcoords="offset points", xytext=(0, offset_y), ha='center', fontsize=6.5,
                    fontweight='bold', color='darkblue', zorder=8,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65))
                    
    ax.set_xlim(xlim_start, xlim_end)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Elevation (deg)", fontsize=11)
    ax.set_xlabel("Time (HST)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    plt.tight_layout()
    fig.savefig(output_alt_png, dpi=180)
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Rotator Angle vs Time Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(f"Rotator Angle vs Time - {date_str} (HST)", fontsize=13, fontweight='bold', pad=10)
    
    # Twilight background
    ax.imshow(img_data, extent=[mdates.date2num(xlim_start), mdates.date2num(xlim_end), -180, 180],
              aspect='auto', origin='lower', zorder=0)
              
    # Rotator limits warning zones
    ax.fill_between([xlim_start, xlim_end], -180, rot_min, color='red', alpha=0.15, zorder=1)
    ax.fill_between([xlim_start, xlim_end], rot_max, 180, color='red', alpha=0.15, zorder=1)
    ax.axhline(rot_min, color='red', linestyle='--', alpha=0.6, label=f'Rot Min ({rot_min}°)')
    ax.axhline(rot_max, color='red', linestyle='--', alpha=0.6, label=f'Rot Max ({rot_max}°)')
    
    for i, row in enumerate(calculated_rows):
        c = colors_list[i % len(colors_list)]
        coord = row["coord"]
        pa_deg = row["pa_deg"]
        
        # Compute continuous rotator trajectory
        pa_traj = observer.parallactic_angle(times_astropy, coord).deg
        rot_traj = (pa_traj + pa_deg + 180.0) % 360.0 - 180.0
        # Split trajectory across +/- 180 wrap discontinuities
        diffs = np.abs(np.diff(rot_traj))
        split_indices = np.where(diffs > 100)[0] + 1
        splits = np.split(np.arange(len(rot_traj)), split_indices)
        for sp in splits:
            ax.plot([times_hst[idx] for idx in sp], rot_traj[sp], color=c, alpha=0.35, linewidth=1.2, zorder=3)
            
        # Highlight active slot
        s_hst = row["t_start_hst"]
        e_hst = row["t_end_hst"]
        ax.plot([s_hst, e_hst], [row["rot_start"], row["rot_end"]], color=c, linewidth=3.5, zorder=6)
        ax.scatter([s_hst, e_hst], [row["rot_start"], row["rot_end"]], color=c, s=35, zorder=7)
        offset_y = 7 if (i % 2 == 0) else -13
        ax.annotate(row["target"], (s_hst, row["rot_start"]),
                    textcoords="offset points", xytext=(0, offset_y), ha='center', fontsize=6.5,
                    fontweight='bold', color='darkblue', zorder=8,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65))
                    
    ax.set_xlim(xlim_start, xlim_end)
    ax.set_ylim(-180, 180)
    ax.set_yticks(range(-180, 181, 60))
    ax.set_ylabel("Rotator Angle (deg)", fontsize=11)
    ax.set_xlabel("Time (HST)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    plt.tight_layout()
    fig.savefig(output_rot_png, dpi=180)
    plt.close(fig)


def build_pdf_report(calculated_rows, date_str, output_alt_png, output_rot_png, output_pdf, config):
    """
    Build landscape A4 PDF report with Page 1 (Plots) and Page 2 (Table).
    """
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(A4),
        rightMargin=1.0*cm, leftMargin=1.0*cm,
        topMargin=1.0*cm, bottomMargin=1.0*cm
    )
    
    constraints = config.get('constraints', {})
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        fontSize=16,
        leading=20,
        fontName='Helvetica-Bold',
        spaceAfter=10
    )
    
    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT
    )
    
    cell_right = ParagraphStyle('CellR', parent=cell_style, alignment=TA_RIGHT)
    cell_center = ParagraphStyle('CellC', parent=cell_style, alignment=TA_CENTER)
    
    header_style = ParagraphStyle(
        'HeaderCell',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.white,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    # =========================================================================
    # Page 1: Graphical Visuals
    # =========================================================================
    elements.append(Paragraph(f"Observation Plan - {date_str} (HST)", title_style))
    elements.append(Spacer(1, 0.4*cm))
    
    img1 = Image(output_alt_png, width=13.6*cm, height=15.5*cm, kind='proportional')
    img2 = Image(output_rot_png, width=13.6*cm, height=15.5*cm, kind='proportional')
    
    t_images = Table([[img1, img2]], colWidths=[13.8*cm, 13.8*cm])
    t_images.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(t_images)
    elements.append(PageBreak())

    # =========================================================================
    # Page 2: Observation Table
    # =========================================================================
    elements.append(Paragraph(f"{date_str} (HST)", title_style))
    elements.append(Spacer(1, 0.5*cm))
    
    # Table headers exactly matching user image:
    headers = [
        "Time (HST)", "LST", "Target", "Air", "Teff",
        "El\n(Start)", "El\n(End)", "Rot\n(Start)", "Rot\n(End)",
        "Moon\nSep", "Moon\nIllum", "Moon\nAlt"
    ]
    
    table_data = [[Paragraph(h.replace("\n", "<br/>"), header_style) for h in headers]]
    bg_commands = []
    warning_color = colors.pink
    
    for row_idx, r in enumerate(calculated_rows):
        row_num = row_idx + 1  # 1-based data row
        
        # Constraint checks
        if 'max_airmass' in constraints and r['airmass'] > constraints['max_airmass']:
            bg_commands.append(('BACKGROUND', (3, row_num), (3, row_num), warning_color))
        if 'min_teff' in constraints and r['teff'] < constraints['min_teff']:
            bg_commands.append(('BACKGROUND', (4, row_num), (4, row_num), warning_color))
        if 'min_altitude' in constraints and (r['el_start'] < constraints['min_altitude'] or r['el_end'] < constraints['min_altitude']):
            bg_commands.append(('BACKGROUND', (5, row_num), (6, row_num), warning_color))
        if 'max_altitude' in constraints and (r['el_start'] > constraints['max_altitude'] or r['el_end'] > constraints['max_altitude']):
            bg_commands.append(('BACKGROUND', (5, row_num), (6, row_num), warning_color))
            
        rot_min = constraints.get('rotator_min', -174.0)
        rot_max = constraints.get('rotator_max', 174.0)
        if r['rot_start'] < rot_min or r['rot_start'] > rot_max or r['rot_end'] < rot_min or r['rot_end'] > rot_max:
            bg_commands.append(('BACKGROUND', (7, row_num), (8, row_num), warning_color))
            
        if 'min_moon_sep' in constraints and r['moon_sep'] < constraints['min_moon_sep']:
            bg_commands.append(('BACKGROUND', (9, row_num), (9, row_num), warning_color))
            
        table_data.append([
            Paragraph(r["time_hst_str"], cell_center),
            Paragraph(r["lst_str"], cell_center),
            Paragraph(r["target"], cell_style),
            Paragraph(f"{r['airmass']:.2f}", cell_right),
            Paragraph(f"{r['teff']:.2f}", cell_right),
            Paragraph(f"{r['el_start']:.1f}", cell_right),
            Paragraph(f"{r['el_end']:.1f}", cell_right),
            Paragraph(f"{r['rot_start']:.1f}", cell_right),
            Paragraph(f"{r['rot_end']:.1f}", cell_right),
            Paragraph(f"{r['moon_sep']:.1f}", cell_right),
            Paragraph(f"{r['moon_illum']:.2f}", cell_right),
            Paragraph(f"{r['moon_alt']:.1f}", cell_right),
        ])
        
    col_widths = [
        2.8 * cm,  # Time
        1.8 * cm,  # LST
        6.5 * cm,  # Target
        1.2 * cm,  # Air
        1.2 * cm,  # Teff
        1.4 * cm,  # El Start
        1.4 * cm,  # El End
        1.8 * cm,  # Rot Start
        1.8 * cm,  # Rot End
        1.6 * cm,  # Moon Sep
        1.6 * cm,  # Moon Illum
        1.6 * cm   # Moon Alt
    ]
    
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    base_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#001f5b')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.whitesmoke]),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    
    t.setStyle(TableStyle(base_style + bg_commands))
    elements.append(t)
    
    doc.build(elements)


def main():
    parser = argparse.ArgumentParser(description="Generate PDF observation plan report from PFS OPE file")
    parser.add_argument("ope", type=str, help="Path to input OPE file (e.g. 2026-11-09_ssp_v2.ope)")
    parser.add_argument("-d", "--pfs-designs-dir", type=str, required=True,
                        help="Path to pfs_designs directory containing pfsDesign-*.fits")
    parser.add_argument("-o", "--output-pdf", type=str, default=None,
                        help="Path to output PDF file (default: obsplan_<date>.pdf)")
    parser.add_argument("-c", "--config", type=str,
                        default=str(Path(__file__).resolve().parent / "config.yaml"),
                        help="Path to config YAML file (default: config.yaml next to this script)")
    parser.add_argument("--slot-duration", type=int, default=20,
                        help="Observation slot duration in minutes (default: 20)")
    args = parser.parse_args()
    
    ope_path = Path(args.ope)
    if not ope_path.exists():
        print(f"Error: OPE file not found: {ope_path}")
        return
        
    designs_dir = Path(args.pfs_designs_dir)
    config = load_config(args.config)
    
    print(f"Parsing OPE file: {ope_path}")
    date_str, params, observations = parse_ope_file(ope_path)
    print(f"  Observation Date: {date_str}")
    print(f"  Observations found: {len(observations)}")
    
    if len(observations) == 0:
        print("Error: No science observations found in OPE file.")
        return
        
    print(f"Indexing pfsDesign files in {designs_dir}...")
    design_map = build_design_index(designs_dir)
    print(f"  Indexed {len(design_map)} entries.")
    
    print("Setting up Subaru observer and models...")
    observer = Observer(location=EarthLocation.of_site(config["location"]["name"]))
    mbm = MoonBrightnessModel()
    
    print("Calculating schedule parameters...")
    calculated_rows = calculate_schedule_parameters(
        observations, params, design_map, observer, mbm, slot_duration_min=args.slot_duration
    )
    print(f"  Calculated parameters for {len(calculated_rows)} targets.")
    
    output_alt_png = Path(f"altitude_vs_time_{date_str}.png")
    output_rot_png = Path(f"rotator_angle_vs_time_{date_str}.png")
    
    print("Generating graphical plots...")
    generate_plots(calculated_rows, date_str, observer, config, output_alt_png, output_rot_png)
    print(f"  Saved {output_alt_png} and {output_rot_png}")
    
    if args.output_pdf:
        output_pdf = Path(args.output_pdf)
    else:
        output_pdf = Path(f"obsplan_{date_str}.pdf")
        
    print(f"Building PDF report: {output_pdf}...")
    build_pdf_report(calculated_rows, date_str, output_alt_png, output_rot_png, output_pdf, config)
    print("PDF build successful!")
    print(f"Report saved to: {output_pdf.resolve()}")


if __name__ == '__main__':
    main()
