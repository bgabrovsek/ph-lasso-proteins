from lasso import _all_lasso_iterator_alphalasso

for i, lasso in enumerate(_all_lasso_iterator_alphalasso(3)):
    print(f"Lasso #{i}", lasso.pdb, lasso.chain, lasso.bridge, lasso.lassoprot_data["symbol"])

