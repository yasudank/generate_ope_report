"""
ope_utils.py
============
Helper functions for generate_ope_report.py:
  - load_config(): load the YAML config and fill in missing keys with defaults
  - MoonBrightnessModel / calculate_teff(): effective exposure time factor (t_eff) model

Extracted from obs_utils.py / make_visibility_map.py in the pfsplan_v3 project.
"""
import os

import numpy as np
import yaml

def load_config(filename='config.yaml'):
    """
    Load observation parameters from a YAML file with sensible defaults.
    """
    defaults = {
        'location': {
            'name': 'Subaru',
            'timezone': 'US/Hawaii',
            'hst_offset_hours': -10
        },
        'twilight': {
            'horizon_deg': -18.0
        },
        'scheduling': {
            'split_margin_minutes': 5,
            'slot_duration_minutes': 20,
            'min_overhead_min': 5,
            'manual_readout_min': 15,
        },
        'constraints': {
            'max_airmass': 1.6,
            'max_airmass_relaxed': 1.85,
            'min_altitude': 32.5,
            'max_altitude': 75.0,
            'min_teff': 0.6,
            'rotator_min': -174.0,
            'rotator_max': 174.0,
            'min_moon_sep': 20.0,
            'max_moon_ill': 1.0,
            'max_moon_alt': 90.0,
            'max_relaxed_count': 8
        },
        'slew': {
            'speed_az': 0.5,
            'speed_el': 0.5,
            'speed_rot': 1.5
        },
        'scheduler': {
            'sa_t0': 500.0,
            'sa_alpha': 0.99999,
            'sa_iterations': 1000000,
            'sa_t_min': 0.5,
            'weight_hard': 1000000.0,
            'weight_split': 1000.0,
            'weight_priority_base': 100.0,
            'weight_teff': 100.0,
            'weight_conn': 0.0,
            'weight_empty': 5000.0,
            'weight_slew': 5.0,
            'max_priority': 4
        }
    }
    
    config = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {filename}: {e}. Using defaults.")
            
    # Merge defaults recursively
    for key, def_val in defaults.items():
        if key not in config:
            config[key] = def_val
        else:
            if isinstance(def_val, dict):
                for k, v in def_val.items():
                    if k not in config[key]:
                        config[key][k] = v
                        
    return config


class MoonBrightnessModel:
    def __init__(self):
        self.k = {
            "g": 0.15, "r": 0.10, "i": 0.09, "z": 0.08, "y": 0.1
        }
        self.Q = {
            "g": 0.129, "r": 0.152, "i": 0.114, "z": 0.048, "y": 0.038
        }
        self.mu_sky = {
            "g": 22.25, "r": 21.18, "i": 20.32, "z": 19.59, "y": 18.28
        }
        self.lam_eff = {
            "g": 478.0, "r": 617.0, "i": 766.0, "z": 888.0, "y": 974.0
        }
        self.Msun = {
            "g": -26.520, "r": -26.922, "i": -27.042, "z": -27.054, "y": -27.059, "V": -26.756
        }

    def X(self, z):
        z = np.radians(z)
        return 1.0 / np.sqrt(1.0 - 0.96 * np.sin(z)**2)

    def tR(self, band, X):
        p = 608.0  # Pressure at Mauna Kea (hPa)
        H = 4.2  # Height of Mauna Kea (km)
        lam = self.lam_eff[band] * 1.0E-03
        tauR = p / 1013.25 * (0.00864 + 6.5E-06 * H) * lam**(-(3.916 + 0.074 * lam + 0.050 / lam))
        return np.exp(-tauR * X)

    def tM(self, band, X):
        lam = self.lam_eff[band] * 1.0E-03
        alpha = -1.38
        kM = np.where(lam < 0.4, 0.050, 0.013 * lam**alpha)
        return 10.0**(-0.4 * kM * X)

    def Bmoon(self, band, alpha, z_moon, X_sky, rho):
        XV = self.Msun[band] - self.Msun["V"]
        phi = 180 - alpha
        Istar = 10.0**(-0.4 * (3.84 + 0.026 * np.abs(phi) + 4.0E-09 * phi**4)) * 10.0**(-0.4 * XV)
        X_moon = self.X(z_moon)
        rho = np.radians(rho)
        fR = 10.0**0.92 * (1.06 + np.cos(rho)**2)
        fM = 10.0**(2.44 - np.degrees(rho) / 40.0)
        BmoonR = fR * Istar * 10.0**(-0.4 * self.k[band] * X_moon) * (1.0 - self.tR(band, X_sky))
        BmoonM = fM * Istar * 10.0**(-0.4 * self.k[band] * X_moon) * (1.0 - self.tM(band, X_sky))
        return BmoonR + BmoonM

    def deltaMag(self, band, alpha, z_moon, z_sky, rho):
        if np.any(z_moon < 90.0):
            X_sky = self.X(z_sky)
            B0 = self.Q[band] * 5.48E+06 * 10.0**(-0.4 * self.mu_sky[band]) * X_sky
            Bm = self.Bmoon(band, alpha, z_moon, X_sky, rho)
            return -2.5 * np.log10((Bm + B0) / B0)
        else:
            return np.zeros_like(z_moon)


def calculate_teff(observer, target_coord, target_alt, target_airmass, moon_coord, moon_altaz, moon_phase_deg, mbm):
    observer_lat = observer.location.lat.deg
    target_dec = target_coord.dec.deg
    
    zmin = abs(target_dec - observer_lat)
    if zmin > 89.9: zmin = 89.9
            
    airmass0 = mbm.X(zmin)
    if airmass0 > 100: airmass0 = 100
    teff0 = 1.0 / (airmass0 * 10**(0.8*mbm.k['r']*(airmass0-1.0)))
    if teff0 == 0: return 0

    airmass = target_airmass    
    z_obs = 90. - target_alt
    z_moon = 90. - moon_altaz.alt.deg
    
    if z_moon >= 90.:
        dmu = 0.0
    else:
        moon_sep = moon_coord.separation(target_coord).deg
        dmu = mbm.deltaMag("r",
                           moon_phase_deg,
                           z_moon,
                           z_obs,
                           moon_sep)

    teff_abs = (1.0 / (10**(-0.4*dmu) * airmass * 10**(0.8*mbm.k['r']*(airmass-1.0))))
    
    return teff_abs / teff0
