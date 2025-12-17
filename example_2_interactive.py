import matplotlib.pyplot as plt

from lasso import get_lasso
from ph import ph_extended_diagrams, bottleneck_dist
from filters import smoothen_and_find_peaks
from plot import plot_diagrams, plot_3D_lasso, interactive_ph_plot
from settings import *


# get a lasso

lasso = get_lasso("8IC0", "F", 0)

# compute PH
ph_diagrams = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"], use_cache=False)
f_bottle = bottleneck_dist(ph_diagrams, BOTTLENECK_MULT_THRESHOLD_PRE)
f_smooth, peaks = smoothen_and_find_peaks(f_bottle, FILTER_WINDOW_SIZE, FILTER_MULT_THRESHOLD, BOTTLENECK_ABS_THRESHOLD_POST)

interactive_ph_plot(lasso, ph_diagrams, "c", f_bottle, f_smooth, peaks)


#plot_diagrams(pers, size=80, show=True)  # Plot persistence diagram