# Installed PA/EHSS CLI validation

The research wheel was installed in a fresh virtual environment without system site packages. All 24 pinned dependency versions and 175 installed numerical source hashes matched the recorded environment. `pip check` passed.

The installed `ccs-xyz` command processed the 92-atom 1AL1 and 49,676-atom 1RYP XYZ inputs with both BVH PA and EHSS. Each used 4,096 incident rays, one fixed scramble, one requested thread and cap 4,096. Every physical result field, including collision histograms and weighted contributions, exactly matched a source-tree replay.

These paired deployment checks are not new independent accuracy data. They validate Windows with the existing Python 3.13 base installation; a different operating system or fresh OS was not tested. Process timings include compilation and are not IMoS speed comparisons.

Wheel SHA256: `90bbfa846a95eac7f7261406501a8ea0ff656dca0d8f758304a803684aefd792`. Evidence: `output/ehss_publication_install_20260913/validation.json`, `design.json`, pinned requirements, build source, wheel and execution logs.

## Reproduction commands

From the workspace root of a fresh extracted source tree (the output directory must not exist), the following sequence recreates the staged build and clean environment. The preparation step records the source environment's exact dependency versions; the retained requirements-pinned.txt is the authoritative lock for this recorded run.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/prepare_ehss_publication_install.py
.venv/Scripts/python.exe -m pip --isolated wheel --no-deps --no-build-isolation --wheel-dir output/ehss_publication_install_20260913/wheels output/ehss_publication_install_20260913/source
.venv/Scripts/python.exe -m venv output/ehss_publication_install_20260913/clean_env
output/ehss_publication_install_20260913/clean_env/Scripts/python.exe -m pip --isolated install --index-url https://pypi.org/simple --only-binary=:all: -r output/ehss_publication_install_20260913/requirements-pinned.txt
.venv/Scripts/python.exe -X utf8 scripts/validate_ehss_publication_install.py
```

The validator installs the staged wheel without modifying the lock, runs pip check, audits installed module origins/hashes, and runs both the installed console command and a source-tree comparison with separate initially empty JIT caches. Existing results are preserved by refusing an existing execution directory. No package was published to PyPI.
