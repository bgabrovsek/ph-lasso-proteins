from lasso import _all_lasso_iterator_alphalasso

from lasso import get_lasso, Lasso
from ph import ph_extended_diagrams, bottleneck_dist
from filters import smoothen_and_find_peaks
from plot import plot_diagrams, plot_3D_lasso, interactive_ph_plot
from settings import *

from filters import get_confusion_stats, find_matchings


def lasso2dict(L: Lasso):
    def get_index(points, p):
        import numpy as np

        idx = np.where((points == p).all(axis=1))[0]
        assert len(idx) == 1, f"Point not found or appears multiple times, times found: {len(idx)}"

        return int(idx[0])

    n_deep_xyz = L.intersections["n-deep-xyz"]
    c_deep_xyz = L.intersections["c-deep-xyz"]
    n_shallow_xyz = L.intersections["n-shallow-xyz"]
    c_shallow_xyz = L.intersections["c-shallow-xyz"]

    n_deep_ind = [get_index(L.Ntail, xyz) for xyz in n_deep_xyz]
    c_deep_ind = [get_index(L.Ctail, xyz) for xyz in c_deep_xyz]
    n_shallow_ind = [get_index(L.Ntail, xyz) for xyz in n_shallow_xyz]
    c_shallow_ind = [get_index(L.Ctail, xyz) for xyz in c_shallow_xyz]

    return {
        "xyz": {"loop": L.loop, "n": L.Ntail, "c": L.Ctail},
        "deep_n": n_deep_ind,
        "deep_c": c_deep_ind,
        "shallow_n": n_shallow_ind,
        "shallow_c": c_shallow_ind,
        "deep_xyz_n": n_deep_xyz,
        "deep_xyz_c": c_deep_xyz,
        "shallow_xyz_n": n_shallow_xyz,
        "shallow_xyz_c": c_shallow_xyz,
    }
    self.pdb = pdb.upper()
    self.chain = chain
    self.ndxes = ndxes
    self.bridge = ndxes[1:3]
    self.endpoints = [ndxes[0], ndxes[3]]
    self.id = '{}_{} {:d}-{:d}'.format(self.pdb, self.chain, *self.bridge)
    self.loop = loop[0]  # xzy
    self.loop_atoms = loop[1]
    self.loop_missing = loop[2]
    self.Ntail = Ntail[0]  # xzy
    self.Ntail_atoms = Ntail[1]
    self.Ntail_missing = Ntail[2]
    self.Ctail = Ctail[0]  # xzy
    self.Ctail_atoms = Ctail[1]
    self.Ctail_missing = Ctail[2]
    self.lassoprot_data = lassoprot_data
    self.intersections = self.xyz_intersections()
    self.n_deep_xyz = self.intersections["n-deep-xyz"]
    self.c_deep_xyz = self.intersections["c-deep-xyz"]

interesting_lassos = [("Q9Y6L6", (142, 463)), ("A9XK88", (188, 232)), ("A0A0M9WNI8", (89, 241)), ("A0A2S6I9K1", (22, 270))]

lassos = list()

for i, lasso in enumerate(_all_lasso_iterator_alphalasso()):

    if (lasso.pdb, lasso.bridge) not in interesting_lassos:
        continue

    lassos.append(lasso)
    #print("**")
    #print(f"Lasso #{i}", lasso.pdb, lasso.chain, lasso.bridge, lasso.lassoprot_data["symbol"])

SL8 = lassos[1]

las = SL8

d = lasso2dict(las)



ph_diagrams = ph_extended_diagrams(d["xyz"]["loop"], d["xyz"]["n"], use_cache=False)
f_bottle = bottleneck_dist(ph_diagrams, BOTTLENECK_MULT_THRESHOLD_PRE)

# TODO: find exact matchings.

import numpy as np
for dist in [0,1,2,3,6,9,12]:
    print("dist", dist)
    for fws in range(0,5):
        for a in np.arange(0.01, 1.2, 0.01):
            for m in np.arange(0.01, 1.2, 0.01):
                #f_smooth, peaks = smoothen_and_find_peaks(f_bottle, FILTER_WINDOW_SIZE//4, FILTER_MULT_THRESHOLD*m, BOTTLENECK_ABS_THRESHOLD_POST*0.1)
                f_smooth, peaks = smoothen_and_find_peaks(f_bottle, fws, FILTER_MULT_THRESHOLD*m, BOTTLENECK_ABS_THRESHOLD_POST*a)
                maxima, maxima_ranges = peaks
                tp, fp, fn = find_matchings(measured_data=maxima_ranges, ground_truth=d["deep_n"], distance=dist)
                #print(tp, fp, fn)
                if len(peaks[0]) == 8 and tp>=7:
                    print(fws, a, m, len(peaks[0]), "confusion", tp, fp, fn)
            #print(m, len(peaks[0]))

print("no!!!")
exit()

fws = 3
a = 0.8
m = 0.55

f_smooth, peaks = smoothen_and_find_peaks(f_bottle, fws, FILTER_MULT_THRESHOLD*m, BOTTLENECK_ABS_THRESHOLD_POST*a)

print("peaks", len(peaks[0]))


interactive_ph_plot(d, ph_diagrams, "n", f_bottle, f_smooth, peaks)



"""


LS_3LS_4 from:
https://www.sciencedirect.com/science/article/abs/pii/S0022283625002839?CMX_ID=&SIS_ID=&dgcid=STMJ_219742_AUTH_SERV_PA&utm_acid=80047596&utm_campaign=STMJ_219742_AUTH_SERV_PA&utm_in=DM568979&utm_medium=email&utm_source=AC_
L_8 - https://alphalasso.cent.uw.edu.pl/view/A9XK88/1/4
L_9 - https://alphalasso.cent.uw.edu.pl/view/A0A2S6I9K1/1/4
L_10 - https://alphalasso.cent.uw.edu.pl/view/A0A0M9WNI8/1/4


Q9Y6L6 F1
LSLS4,3 
142-463
N-termini +47, +59, -80, +93
C-termini -470, +552, +576

A9XK88 F1
L+8N 
188-232
N-termini +59, -73, +88, -102, +112, -133, +148, -175

A0A0M9WNI8 F1
L+10C
89-241
C-termini +275, -294, +309, -325, +336, -391, +404, -444, +458, -468

A0A2S6I9K1 F1
L-9C 
Loop 22-270
C termini: -299, +313, -382, +394, -434, +447, -457, +470, -500


"""