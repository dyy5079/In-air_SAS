import os
import h5py
import numpy as np
import re

def fileAttribute(filename):
    # Pattern: t<digit(s)>e<digit(s)>_<digit(s)>
    pattern = r't(\d+)e(\d+)_(\d+).h5'
    match = re.match(pattern, filename)

    if match:
        t = int(match.group(1))
        e = int(match.group(2))
        trial = int(match.group(3))
        
        # Customize these descriptions based on your experiment
        target = {
            0: "No Targets",
            1: "Solid Sphere",
            2: "Hollow Sphere",  # Customize this
            3: "Letter O",
            4: "Letter Q",
        }
        
        env = {
            1: "Free Field",  # Customize this
            2: "Flat Interface",
            3: "Rough Interface",
            4: "Partially Buried in Rough Interface",
        }
        
        targetDC = target.get(t, f"Target Type {t}")
        envDC = env.get(e, f"Experiment {e}")

        return {
            'target': targetDC,
            'env': envDC,
            'trial': trial,
        }
    else:
        return None

def saveh5(A, filename=None, output_dir=None, channel=None):

    if output_dir and filename and channel is not None:
        xVec = A.Results.Bp.xVect
        yVec = A.Results.Bp.yVect
        image = A.Results.Bp.image

        file = fileAttribute(filename)
    
        complex_output_path = os.path.join(output_dir, f'{filename[:-3]}_ch{channel}_tsRC.h5')
        with h5py.File(complex_output_path, 'w') as hf:
            # Store complex image chips
            hf.create_dataset('tsRC', data=np.array(image), dtype=np.complex64)
            
            # Store metadata for MATLAB compatibility
            if file:
                # Store as byte strings for MATLAB compatibility
                hf.create_dataset('target', data=np.bytes_(file['target']))
                hf.create_dataset('env', data=np.bytes_(file['env']))
                hf.create_dataset('trial', data=file['trial'], dtype=np.int32)

    else:
        print("Output directory, filename, or channel is not specified. Skipping save.")
        return