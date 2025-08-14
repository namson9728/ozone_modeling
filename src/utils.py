import numpy as np

def run_AM(script_path, outpath, airmass, ZA, nscale, za_units, freq_start, freq_end, script=None):
    print(f"Starting am for AM={airmass:.3f} (ZA={np.degrees(ZA):.1f} deg); nscale={nscale:.2f}")
    if script is None:
        script = f"""
        /usr/local/bin/am MaunaKea_SON_50.amc \
        {freq_start} GHz {freq_end} GHz 1 MHz {ZA} {za_units} 277 K {10**nscale} \
        > {outpath}MaunaKea_Tb_Spectrum_{airmass:.3f}_{nscale:.2f}.out \
        2> {outpath}MaunaKea_Tb_Spectrum_{airmass:.3f}_{nscale:.2f}.err
        """
    import os
    import subprocess
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")    # Add the shebang line
        f.write(script)
    os.chmod(script_path, 0o755)       # Make the script executable

    subprocess.call(f'./{script_path}')

def run_AM_batch(script_path, outpath, airmass_map, za_map, nscale_map, freq_start, freq_end, za_units, script=None):
    for ZA, airmass in zip(za_map, airmass_map):
        for nscale in nscale_map:
            run_AM(script_path, outpath, airmass, ZA, nscale, za_units, freq_start, freq_end, script)