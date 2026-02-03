"""
Computes the accuracy of the PH method on the proteins from PDB, namely on the LassoProt website.
"""

from lasso import all_lasso_iterator
from filters import get_confusion_stats, compute_statistics_from_confusion
from ph import ph_extended_diagrams, bottleneck_dist
from itertools import product
from pathlib import Path
import pickle

from settings import *

NUM_PROT = 4847
distances = [0, 1, 3, 6, 9, 12, 15, 18]

confusion = {d:[] for d in distances}
quality = {d:[] for d in distances}

path = "accuracy_pdb.pkl"

"""
0 -> {'f1': 0.5288796493988088, 'precision': 0.4626462203872997, 'recall': 0.6172459016393442, 'intersection_quality': 0.6157116692830978, 'all_intersections': 15288, 'positives': 9413, 'false_positives': 10933, 'false_negatives': 5837}
1 -> {'f1': 0.5778909945973182, 'precision': 0.5073233067924899, 'recall': 0.6712622748260388, 'intersection_quality': 0.6696075251378527, 'all_intersections': 15415, 'positives': 10322, 'false_positives': 10024, 'false_negatives': 5055}
3 -> {'f1': 0.6523710193297175, 'precision': 0.5764277990759854, 'recall': 0.7513613940675251, 'intersection_quality': 0.7517948717948718, 'all_intersections': 15600, 'positives': 11728, 'false_positives': 8618, 'false_negatives': 3881}
6 -> {'f1': 0.7286855883163406, 'precision': 0.6498574658409515, 'recall': 0.8292774711490216, 'intersection_quality': 0.8408267090620032, 'all_intersections': 15725, 'positives': 13222, 'false_positives': 7124, 'false_negatives': 2722}
9 -> {'f1': 0.7751057731677358, 'precision': 0.6978275828172614, 'recall': 0.8716311621339554, 'intersection_quality': 0.903008331743306, 'all_intersections': 15723, 'positives': 14198, 'false_positives': 6148, 'false_negatives': 2091}
12 -> {'f1': 0.7875990878488434, 'precision': 0.7129656935024083, 'recall': 0.8796846573681019, 'intersection_quality': 0.9257179323548181, 'all_intersections': 15670, 'positives': 14506, 'false_positives': 5840, 'false_negatives': 1984}
15 -> {'f1': 0.8025196511252288, 'precision': 0.7326255775090927, 'recall': 0.8871562909177478, 'intersection_quality': 0.9605000322185708, 'all_intersections': 15519, 'positives': 14906, 'false_positives': 5440, 'false_negatives': 1896}
18 -> {'f1': 0.812781954887218, 'precision': 0.7438317113929028, 'recall': 0.8958210015390079, 'intersection_quality': 0.9775854272979781, 'all_intersections': 15481, 'positives': 15134, 'false_positives': 5212, 'false_negatives': 1760}

"""



for counter, lasso in enumerate(all_lasso_iterator(include_trivial=True)):


    print()
    print("***", counter, "***", round(100 * counter / NUM_PROT,2), "%")
    print(lasso["id"])
    print("Loop", lasso["xyz"]["loop"].shape, "Tails:", lasso["xyz"]["c"].shape, lasso["xyz"]["n"].shape)


    ph_diagrams_c = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"], use_cache=True)
    ph_diagrams_n = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["n"], use_cache=True)

    #print(ph_diagrams_c[0].shape, len(ph_diagrams_c[1]), "and", ph_diagrams_n[0].shape, len(ph_diagrams_n[1]))

    f_bottle_c = bottleneck_dist(ph_diagrams_c, BOTTLENECK_MULT_THRESHOLD_PRE)
    f_bottle_n = bottleneck_dist(ph_diagrams_n, BOTTLENECK_MULT_THRESHOLD_PRE)


    for d in distances:
        confusion[d].append(get_confusion_stats(
            bottle=f_bottle_c,
            deep=lasso["deep_c"],
            shallow=lasso["shallow_c"],
            threhold_abs=BOTTLENECK_ABS_THRESHOLD_POST,
            window=FILTER_WINDOW_SIZE,
            threshold_rel=FILTER_MULT_THRESHOLD,
            ignore_non_lassos=True,
            atom_distance=d
        ))
        confusion[d].append(get_confusion_stats(
            bottle=f_bottle_n,
            deep=lasso["deep_n"],
            shallow=lasso["shallow_n"],
            threhold_abs=BOTTLENECK_ABS_THRESHOLD_POST,
            window=FILTER_WINDOW_SIZE,
            threshold_rel=FILTER_MULT_THRESHOLD,
            ignore_non_lassos=True,
            atom_distance=d
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

