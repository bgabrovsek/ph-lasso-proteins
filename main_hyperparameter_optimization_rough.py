"""
Figures out which parameters (thresholds, window size,...) produce best results (based on 1000 lassos).
Should be used before doing final computations. Results should be applied to constants in settings.py
"""

SKIP_EVERY = 20
SAMPLE_SIZE = 1000

from lasso import all_lasso_iterator
from filters import get_confusion_stats, compute_statistics_from_confusion
from ph import ph_extended_diagrams, bottleneck_dist
from settings import *
from itertools import product
from pathlib import Path
import pickle

# bottle_mult_thr_values = [0.2, 0.3, 0.5, 0.8]
# bottle_abs_thr_values = [0.005, 0.01, 0.025]
# filter_win_values = [2,3,5]
# filter_thr_values = [0.5]

"""
BEST RESULTS
(0.3, 0.01, 2, 0.5) -> {'f1': 0.7844756399669695, 'precision': 0.7274119448698315, 'recall': 0.8512544802867383, 'intersection_quality': 0.8715596330275229, 'all_intersections': 545, 'positives': 475, 'false_positives': 178, 'false_negatives': 83}
(0.3, 0.025, 2, 0.5) -> {'f1': 0.7838283828382837, 'precision': 0.7274119448698315, 'recall': 0.8497316636851521, 'intersection_quality': 0.8715596330275229, 'all_intersections': 545, 'positives': 475, 'false_positives': 178, 'false_negatives': 84}
(0.3, 0.005, 2, 0.5) -> {'f1': 0.7834710743801653, 'precision': 0.7269938650306749, 'recall': 0.8494623655913979, 'intersection_quality': 0.8697247706422019, 'all_intersections': 545, 'positives': 474, 'false_positives': 178, 'false_negatives': 84}
(0.3, 0.005, 3, 0.5) -> {'f1': 0.7651858567543065, 'precision': 0.7603603603603604, 'recall': 0.7700729927007299, 'intersection_quality': 0.7829313543599258, 'all_intersections': 539, 'positives': 422, 'false_positives': 133, 'false_negatives': 126}
(0.3, 0.01, 3, 0.5) -> {'f1': 0.7651858567543065, 'precision': 0.7603603603603604, 'recall': 0.7700729927007299, 'intersection_quality': 0.7829313543599258, 'all_intersections': 539, 'positives': 422, 'false_positives': 133, 'false_negatives': 126}
(0.3, 0.025, 3, 0.5) -> {'f1': 0.7651858567543065, 'precision': 0.7603603603603604, 'recall': 0.7700729927007299, 'intersection_quality': 0.7829313543599258, 'all_intersections': 539, 'positives': 422, 'false_positives': 133, 'false_negatives': 126}
(0.5, 0.01, 2, 0.5) -> {'f1': 0.734130634774609, 'precision': 0.7430167597765364, 'recall': 0.7254545454545455, 'intersection_quality': 0.7361623616236163, 'all_intersections': 542, 'positives': 399, 'false_positives': 138, 'false_negatives': 151}
(0.5, 0.025, 2, 0.5) -> {'f1': 0.734130634774609, 'precision': 0.7430167597765364, 'recall': 0.7254545454545455, 'intersection_quality': 0.7361623616236163, 'all_intersections': 542, 'positives': 399, 'false_positives': 138, 'false_negatives': 151}
(0.5, 0.005, 2, 0.5) -> {'f1': 0.7329650092081031, 'precision': 0.7425373134328358, 'recall': 0.7236363636363636, 'intersection_quality': 0.7343173431734318, 'all_intersections': 542, 'positives': 398, 'false_positives': 138, 'false_negatives': 152}
(0.3, 0.005, 5, 0.5) -> {'f1': 0.7221621621621621, 'precision': 0.8106796116504854, 'recall': 0.6510721247563352, 'intersection_quality': 0.6423076923076924, 'all_intersections': 520, 'positives': 334, 'false_positives': 78, 'false_negatives': 179}

(0.3, 0.01, 2, 0.5) -> {'f1': 0.7844756399669695, 'precision': 0.7274119448698315, 'recall': 0.8512544802867383, 'intersection_quality': 0.8715596330275229, 'all_intersections': 545, 'positives': 475, 'false_positives': 178, 'false_negatives': 83}

"""
#(0.3, 0.01, 2, 0.5)
bottle_mult_thr_values = [0.25, 0.3, 0.35]
bottle_abs_thr_values = [0.0075, 0.01, 0.015, 0.02 ]
filter_win_values = [2,3,4]
filter_thr_values = [0.5, 0.6, 0.4]

# 0.3 OK
# (0.3, 0.075, 2, 0.5)

bottle_mult_thr_values = [0.3]
bottle_abs_thr_values = [0.0075]
filter_win_values = [2,3,4]
filter_thr_values = [0.5]

quality = {}
confusion = {}



path = Path("confusion_rough_8.pkl")

if path.exists():
    with path.open("rb") as f:
        confusion = pickle.load(f)
else:



    # for counter, (b_mult, b_abs, f_win, f_thr) in enumerate):

        # print()
        # print("****", counter, "/", all, "****", "(", counter / all * 100, "%)")
        # print()
        # confusion_stats = []

    for counter, lasso in enumerate(all_lasso_iterator(include_trivial=True)):
        if counter % SKIP_EVERY != 0: continue
        if counter // SKIP_EVERY >= SAMPLE_SIZE: break

        print()
        print("***", counter + 1, "/", SAMPLE_SIZE, "***", 100*(counter // SKIP_EVERY)/SAMPLE_SIZE, "%")
        print(lasso["id"])
        print()

        print("Loop", lasso["xyz"]["loop"].shape, "Tails:", lasso["xyz"]["c"].shape, lasso["xyz"]["n"].shape)


        ph_diagrams_c = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"], use_cache=True)
        ph_diagrams_n = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["n"], use_cache=True)

        #print(ph_diagrams_c[0].shape, len(ph_diagrams_c[1]), "and", ph_diagrams_n[0].shape, len(ph_diagrams_n[1]))

        for bot_rel in bottle_mult_thr_values:
            f_bottle_c = bottleneck_dist(ph_diagrams_c, bot_rel) # TODO: this could be optimized
            f_bottle_n = bottleneck_dist(ph_diagrams_n, bot_rel) # TODO: this could be optimized

            for bot_abs, win, filt_thr in product(bottle_abs_thr_values, filter_win_values, filter_thr_values):
                key = (bot_rel, bot_abs, win, filt_thr)

                if key not in confusion: confusion[key] = []

                confusion[key].append(get_confusion_stats(
                    bottle=f_bottle_c,
                    deep=lasso["deep_c"],
                    shallow=lasso["shallow_c"],
                    threhold_abs=bot_abs,
                    window=win,
                    threshold_rel=filt_thr,
                    ignore_non_lassos=True,
                    atom_distance=9
                ))

                confusion[key].append(get_confusion_stats(
                    bottle=f_bottle_n,
                    deep=lasso["deep_n"],
                    shallow=lasso["shallow_n"],
                    threhold_abs=bot_abs,
                    window=win,
                    threshold_rel=filt_thr,
                    ignore_non_lassos=True,
                    atom_distance=9
                ))



with open(path, "wb") as f:
    pickle.dump(confusion, f)

print()
print("COMPUTING QUALITY")
print()

for key in confusion:
    quality[key] = compute_statistics_from_confusion(confusion[key])

for k, v in list(quality.items()):
    print(k, "->", v)
print("...")

print()
print("BEST RESULTS")
top = sorted(
    quality.items(),
    key=lambda item: item[1]['f1'],
    reverse=True
)

for k, v in top:
    print(k, "->", v)
