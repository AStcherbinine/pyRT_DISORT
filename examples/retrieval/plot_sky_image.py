"""
Sky Image
=========

Simulate an image of the sky taken by the Mastcam-Z instrument on Mars2020.

"""

# %%
# First import everything needed for this example.
import matplotlib.pyplot as plt
import numpy as np
import disort
import pyrt

# %%
# Before we simulate the sky, it's useful to take a look at what variables we
# need. Many variables must be either floats, ints, or booleans, but some
# need to be arrays that have a specific shape.
print(disort.disort.__doc__)
