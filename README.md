# PFS OPE Report Generator

Reads a PFS OPE file and its pfsDesign files and writes a PDF observation plan report.

- **Page 1:** Elevation vs. time (HST) and instrument rotator angle vs. time (HST)
- **Page 2:** Observation table (Time, LST, Target, Airmass, t_eff, El, Rot, Moon). Values that break a limit in `config.yaml` are highlighted.

## Files

| File | Contents |
|---|---|
| `generate_ope_report.py` | Main script |
| `ope_utils.py` | Config loader and the Moon sky-brightness / t_eff model |
| `config.yaml` | Site, twilight and constraint settings |
| `requirements.txt` | Python dependencies |
| `setup_venv.sh` | Creates the virtual environment |

## Setup

You need Python 3.9 or later. Python 3.12 has been tested.

```bash
./setup_venv.sh            # creates .venv/ and installs requirements.txt
source .venv/bin/activate
```

To set it up by hand instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python generate_ope_report.py <ope_file> -d <pfs_designs_dir> [-o output.pdf] [-c config.yaml] [--slot-duration 20]
```

| Option | Description |
|---|---|
| `ope_file` | Input OPE file, e.g. `2026-11-09_ssp_v2.ope` |
| `-d, --pfs-designs-dir` | **Required.** Directory with the `pfsDesign-*.fits` files |
| `-o, --output-pdf` | Output PDF. Default: `obsplan_<date>.pdf` |
| `-c, --config` | Config YAML. Default: the `config.yaml` next to the script |
| `--slot-duration` | Observation slot length in minutes. Default: 20 |

The script writes these files to the current directory:

- `obsplan_<date>.pdf`
- `altitude_vs_time_<date>.png`
- `rotator_angle_vs_time_<date>.png`

## Notes

- `astropy` fetches site and IERS data from the internet the first time it runs, and caches them under `~/.astropy/cache`. After that the script works offline.
- Settings missing from `config.yaml` fall back to the defaults in `ope_utils.load_config()`.
