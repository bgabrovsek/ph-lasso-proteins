SKIP_EVERY = 10
SAMPLE_SIZE = 100

from lasso import all_lasso_iterator
from filters import get_confusion_stats, compute_statistics_from_confusion
from ph import ph_extended_diagrams, bottleneck_dist

from settings import *

confusion_stats = []

for counter, lasso in enumerate(all_lasso_iterator(include_trivial=True)):
    if counter % SKIP_EVERY != 0: continue
    if counter // SKIP_EVERY >= SAMPLE_SIZE: break

    ph_diagrams = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"], use_cache=True)
    f_bottle = bottleneck_dist(ph_diagrams, BOTTLENECK_MULT_THRESHOLD_PRE)

    for q in "cn":

        confusion_stats.append(get_confusion_stats(
            bottle=f_bottle,
            deep=lasso["deep_" + q],
            shallow=lasso["shallow_" + q],
            threhold_abs=BOTTLENECK_ABS_THRESHOLD_POST,
            window=FILTER_WINDOW_SIZE,
            threshold_rel=FILTER_MULT_THRESHOLD,
            ignore_non_lassos=IGNORE_NON_LASSOS,
            atom_distance=9
        ))

quality = compute_statistics_from_confusion(confusion_stats)
print(quality)