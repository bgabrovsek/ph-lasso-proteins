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



8.3

0 -> {'f1': 0.5283939280897587, 'precision': 0.4631753697555086, 'recall': 0.6149889786921381, 'intersection_quality': 0.6134728144989339, 'all_intersections': 15008, 'positives': 9207, 'false_positives': 10671, 'false_negatives': 5764}
1 -> {'f1': 0.5772364696800756, 'precision': 0.5078478720193178, 'recall': 0.6685873236638188, 'intersection_quality': 0.6669529598308668, 'all_intersections': 15136, 'positives': 10095, 'false_positives': 9783, 'false_negatives': 5004}
3 -> {'f1': 0.6533568101783482, 'precision': 0.5786799476808532, 'recall': 0.7501630363897221, 'intersection_quality': 0.7506525711302532, 'all_intersections': 15324, 'positives': 11503, 'false_positives': 8375, 'false_negatives': 3831}
6 -> {'f1': 0.7302239225835491, 'precision': 0.6529328906328604, 'recall': 0.8282705807275048, 'intersection_quality': 0.8390871476596845, 'all_intersections': 15468, 'positives': 12979, 'false_positives': 6899, 'false_negatives': 2691}
9 -> {'f1': 0.7747170021747616, 'precision': 0.698913371566556, 'recall': 0.8689642231673755, 'intersection_quality': 0.8984673090603376, 'all_intersections': 15463, 'positives': 13893, 'false_positives': 5985, 'false_negatives': 2095}
12 -> {'f1': 0.7874234064380181, 'precision': 0.7143575812455981, 'recall': 0.8771387979492248, 'intersection_quality': 0.9220180507759237, 'all_intersections': 15401, 'positives': 14200, 'false_positives': 5678, 'false_negatives': 1989}
15 -> {'f1': 0.8031231957770875, 'precision': 0.734782171244592, 'recall': 0.8854804486207942, 'intersection_quality': 0.9579589427428347, 'all_intersections': 15247, 'positives': 14606, 'false_positives': 5272, 'false_negatives': 1889}
18 -> {'f1': 0.8133501535761298, 'precision': 0.746000603682463, 'recall': 0.8940672856626071, 'intersection_quality': 0.9750789058390321, 'all_intersections': 15208, 'positives': 14829, 'false_positives': 5049, 'false_negatives': 1757}

"""

from collections import defaultdict
lasso_types = defaultdict(int)

for counter, lasso in enumerate(all_lasso_iterator(include_trivial=True)):
    lasso_types[lasso["symbol"]] += 1

    #
    #
    # print()
    # print("***", counter, "***", round(100 * counter / NUM_PROT,2), "%")
    # print(lasso["id"])
    # print("Loop", lasso["xyz"]["loop"].shape, "Tails:", lasso["xyz"]["c"].shape, lasso["xyz"]["n"].shape)
    #
    #
    # ph_diagrams_c = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"], use_cache=True)
    # ph_diagrams_n = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["n"], use_cache=True)
    #
    # #print(ph_diagrams_c[0].shape, len(ph_diagrams_c[1]), "and", ph_diagrams_n[0].shape, len(ph_diagrams_n[1]))
    #
    # f_bottle_c = bottleneck_dist(ph_diagrams_c, BOTTLENECK_MULT_THRESHOLD_PRE)
    # f_bottle_n = bottleneck_dist(ph_diagrams_n, BOTTLENECK_MULT_THRESHOLD_PRE)
    #
    #
    # for d in distances:
    #     confusion[d].append(get_confusion_stats(
    #         bottle=f_bottle_c,
    #         deep=lasso["deep_c"],
    #         shallow=lasso["shallow_c"],
    #         threhold_abs=BOTTLENECK_ABS_THRESHOLD_POST,
    #         window=FILTER_WINDOW_SIZE,
    #         threshold_rel=FILTER_MULT_THRESHOLD,
    #         ignore_non_lassos=True,
    #         atom_distance=d
    #     ))
    #     confusion[d].append(get_confusion_stats(
    #         bottle=f_bottle_n,
    #         deep=lasso["deep_n"],
    #         shallow=lasso["shallow_n"],
    #         threhold_abs=BOTTLENECK_ABS_THRESHOLD_POST,
    #         window=FILTER_WINDOW_SIZE,
    #         threshold_rel=FILTER_MULT_THRESHOLD,
    #         ignore_non_lassos=True,
    #         atom_distance=d
    #     ))


print("Lasos types:")
for k, v in sorted(lasso_types.items()):
    print(k, "->", v)

exit()

with open(path, "wb") as f:
    pickle.dump(confusion, f)

print()
print("COMPUTING QUALITY")
print()

for key in confusion:
    quality[key] = compute_statistics_from_confusion(confusion[key])

for k, v in list(quality.items()):
    print(k, "->", v)

