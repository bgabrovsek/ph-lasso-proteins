import matplotlib.pyplot as plt

from lasso import get_lasso
from ph import ph_extended_diagrams
from plot import plot_diagrams, plot_3D_lasso

# get a lasso

lasso = get_lasso("8IC0", "F", 0)
#lasso = get_lasso("8OH9", "E", 0)


print(lasso.keys())
# compute PH
pers = ph_extended_diagrams(lasso["xyz"]["loop"], lasso["xyz"]["c"])


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

#ground_intersections = lasso["deep_xyz_n"]
print(lasso["deep_n"])
print(lasso["deep_c"])

plot_3D_lasso(
    tailN=lasso["xyz"]["n"],
    loop=lasso["xyz"]["loop"],
    tailC=lasso["xyz"]["c"],
    deep_xyz=[lasso["xyz"]["c"][i] for i in lasso["deep_c"]] + [lasso["xyz"]["n"][i] for i in lasso["deep_n"]],
    terminus="c",
    current_atom_index=None,
    ax=ax
)

ax.set_title("3D Lasso")
plt.tight_layout()
plt.show()


#plot_diagrams(pers, size=80, show=True)  # Plot persistence diagram