"""
Precomputes all PH diagrams for lassos in LassoProt. Can be called before doing computations, to speed things up.
"""
import sys
from lasso import all_lasso_iterator
from ph import ph_extended_diagrams

arg = sys.argv[1] if len(sys.argv) > 1 else None

for lasso in all_lasso_iterator(include_trivial=True, pdb_starts_with=arg):
    print(lasso["id"])
    try:
        pers_n = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["n"])
        pers_c = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"])
    except:
        print()
        print("Problem:", lasso["id"])
        print()