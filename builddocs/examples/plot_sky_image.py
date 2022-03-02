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
# need.
print(disort.disort.__doc__)
