# Legacy / archived

Files here are superseded by the BRZRKR-themed app at the project root.
They are kept for reference and easy recovery — nothing here is
imported by the live codebase.

| File | Replaced by |
| --- | --- |
| `dashboard.py` (Streamlit browser dashboard) | `BRZRKR.py` |
| `desktop_app.py` (CustomTkinter dashboard) | `BRZRKR.py` + `brzrkr_app/` |
| `launcher.command` (Dock launcher for desktop_app) | `BRZRKR.command` |
| `build_macapp.sh` (PyInstaller wrapper for desktop_app) | `build_brzrkr.sh` |

To delete entirely:

```bash
rm -rf _legacy/
```
