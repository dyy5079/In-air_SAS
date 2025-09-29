import numpy as np
import matplotlib.pyplot as plt
import os
from .sasColormap import sasColormap
from matplotlib.colors import ListedColormap

def plotSasImage(A, dynamicRange, normFlag=None, output_dir='output_plots', filename='sas_image', chanSelect=[1], m=0):
    """
    PLOTSASIMAGE Makes plot of linear AirSAS image
        A = Single-channel AirSAS data structure
        dynamicRange = dynamic range of the image to display
        normFlag = range-dependent normalization flag (1=apply range
        normalization, 0=no normalization)
    """
    if normFlag is None:
        normFlag = 1

    xVect = A.Results.Bp.xVect
    yVect = A.Results.Bp.yVect
    image = A.Results.Bp.image

    if normFlag:
        rNorm = 20 * np.log10(np.tile(yVect, (len(xVect), 1)).T)
    else:
        rNorm = 0

    img = 20 * np.log10(np.abs(image) + 1e-12) + rNorm #adding 1e-12 to avoid log of zero

    plt.imshow(
        img,
        extent=[xVect[0], xVect[-1], yVect[-1], yVect[0]],
        aspect='auto',
        cmap=ListedColormap(sasColormap())
    )
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    clim = [max(img.max() - dynamicRange, img.min()), img.max()]
    plt.clim(clim)
    plt.xlabel('Along Track (m)')
    plt.ylabel('Cross Track (m)')
    plt.axis('image')
    h = plt.colorbar()
    if normFlag:
        h.set_label('Amplitude (dB re: 1V @ 1m)')
    else:
        h.set_label('Amplitude (dB re: 1V)')
    #plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_ch{chanSelect[m]}_Backprojection_ch{m}.png'), dpi=300, bbox_inches='tight')
    plt.gca().invert_yaxis()  # 0 at top for y-axis
    plt.show()