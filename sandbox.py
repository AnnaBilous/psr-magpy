from astropy.io import fits
import numpy as np

col1 = fits.Column(
    name="var",
    format="PE",
    array=np.array([[45.0, np.exp(1)], [11, 12, 1.3]], dtype=np.object_),
)
col2 = fits.Column(name="xyz", format="2D(1000)", array=[[11, 3], [12, np.pi]])
hdu = fits.BinTableHDU.from_columns([col1])  # ], col2])
data = hdu.data
hdu.writeto("variable_length_table.fits", overwrite=True)
with fits.open("variable_length_table.fits") as hdul:
    print(repr(hdul[1].header))
    print(hdul[1].data)
    print("%e" % (hdul[1].data[0]))
