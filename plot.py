import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

#from old_lassopdb import get_terminus_data
#from old_bottleneck import bottleneck_dist, smooth, smooth_and_threshold, find_maxima_with_ranges
#from old_cached_ph import ph_lasso



def plot_3D_lasso(tailN, loop, tailC, deep_xyz, terminus, current_atom_index, ax):
    """Plots a 3D image of the lasso with tails, plots also deep intersections and an colors an additional atom.

    Args:
        tailN: xyz coors of the N tail
        loop:  xys of the loop
        tailC: xyz C tail
        deep_xyz: coords of deep intersections
        terminus: current tail, can be "N" or "C"
        current_atom: color a special atom different colot
        ax: matplotlib axis
    Returns:
        the scatter used to plot the current atom
    """

    if tailN is not None:
        ax.scatter3D(*np.transpose(tailN), s=40, alpha=0.40, color="tab:blue")
    if tailC is not None:
        ax.scatter3D(*np.transpose(tailC), s=40, alpha=0.40, color="tab:red")
    if loop is not None:
        ax.scatter3D(*np.transpose(loop), s=40, alpha=0.7, color="gray")
    if tailN is not None:
        ax.plot(*np.transpose(tailN), alpha=0.40, c="tab:blue")
    if tailC is not None:
        ax.plot(*np.transpose(tailC), alpha=0.40, c="tab:red")
    if loop  is not None:
        ax.plot(*np.transpose(loop), alpha=0.80, c="gray")

    tail_xyz = tailN if terminus in "Nn" else tailC  # get tail coords

    if deep_xyz is not None and len(deep_xyz):
        scatter = ax.scatter3D(*np.transpose(deep_xyz), s=60, alpha=1.0, color="tab:green")  # intersections

    if current_atom_index is not None:
        scatter = ax.scatter3D(*(tail_xyz[current_atom_index]), s=60, alpha=1.0, color="black")  # current point
    else:
        scatter = None

    ax.set_title(f"Atom {current_atom_index}")

    return scatter

def draw_cost_function(f, compl_f, indices, ax, for_paper =False):
    """ plots a cost function of a PH diagram. Also draw a transparent compl_function. Mark the points contained in indices.
    Args:
        f: y's of a function
        compl_f: y's of a complementary function (can be None)
        indices: tuple of special indices (x's) of f:
            - indices of shallow points, indices of deep points, indices of PH intersection, current atom index
        ax: matpltolib axis

    Returns: scatter of the current index
    """
    shallow_index, deep_index, ph_intersections, current_atom_index = indices

    x = np.arange(len(f))
    if for_paper:
        ax.plot(x, compl_f, color='tab:blue', alpha=1.0, linewidth=2)  # plot 0th homology
        legend = ["bottleneck"]
    else:
        ax.plot(x, compl_f, color='tab:blue', alpha=0.4)  # plot 0th homology
        ax.plot(x, f, color='tab:blue', label='H_0')  # plot 0th homology
        legend = ["bottleneck", "smoothened"]

    # plot position of "theoretical" minimal-surface intersection
    if for_paper:

        if shallow_index is not None:
            ax.scatter(x[shallow_index], compl_f[shallow_index], color="tab:green", alpha=0.3)
            legend.append("shallow")
        if deep_index is not None:
            ax.scatter(x[deep_index], compl_f[deep_index], color="tab:green")
            legend.append("deep")
        if ph_intersections is not None:
            ax.scatter(x[ph_intersections], compl_f[ph_intersections], color="tab:red")
            legend.append("max")
    else:

        if shallow_index is not None:
            ax.scatter(x[shallow_index], f[shallow_index], color="tab:green", alpha=0.3)
            legend.append("shallow")
        if deep_index is not None:
            ax.scatter(x[deep_index], f[deep_index], color="tab:green")
            legend.append("deep")
        if ph_intersections is not None:
            ax.scatter(x[ph_intersections], f[ph_intersections], color="tab:red")
            legend.append("max")

    if current_atom_index is not None and for_paper is False:
        scatter_cost = ax.scatter(x[current_atom_index], f[current_atom_index], color="black")  # current slider position
    else:
        scatter_cost = None
    ax.legend(loc='lower left')
    ax.legend(legend, loc='lower left')

    plt.xlabel("Atom in the C-tail", fontsize=14)  # Change x-axis label size
    plt.ylabel("Bottleneck distance", fontsize=14)  # Change y-axis label size
    plt.xticks(fontsize=12)  # Change x-axis tick labels size
    plt.yticks(fontsize=12)  # Change y-axis tick labels size
    return scatter_cost


def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams.

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams,
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram.
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes.
        Some options are 'default', 'seaborn', 'sequential'.
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type.
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death.
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()



def interactive_ph_plot(lasso,
                        ph_diagrams,
                        terminus,
                        f_bottle,
                        f_smooth,
                        peaks,
                        for_paper=False):
    """
      Generates an interactive plot displaying:
      1. A 3D visualization of a protein lasso motif.
      2. A persistence homology (PH) diagram.
      3. A cost function graph derived from PH.

      The visualization includes a **slider** that allows users to interactively
      navigate through atoms in the tail of the protein and observe changes in
      their PH properties.

      Parameters
      ----------
      lasso : Lasso
          An instance of the Lasso class representing the lasso motif in the protein.
      terminus : str
          Either "N" or "C" specifying whether to analyze the N-terminus or C-terminus.
      threshold_mult_bottleneck : float, optional
          Multiplication factor for bottleneck distance thresholding (default: 0.3).
      filter_window : int, optional
          Window size for smoothing and filtering the cost function (default: 6).
      threshold_mult_filtered : float, optional
          Threshold multiplier for selecting relevant points in the cost function (default: 0.1).

      Returns
      -------
      None
          The function generates an interactive visualization but does not return any values.
      """

    # get loop coords & tail data
    tailN, loop, tailC = lasso["xyz"]["n"], lasso["xyz"]["loop"], lasso["xyz"]["c"]

    shallow_indices_n, deep_indices_n = lasso["shallow_n"], lasso["deep_n"]
    shallow_indices_c, deep_indices_c = lasso["shallow_c"], lasso["deep_c"]

    deep_xyz_n =[tailN[i] for i in deep_indices_n]
    deep_xyz_c =[tailC[i] for i in deep_indices_c]

    deep_xyz_n_2 = lasso["deep_xyz_n"]
    deep_xyz_c_2 = lasso["deep_xyz_c"]

    print("Deep N:", deep_xyz_n, deep_xyz_n_2)
    print("Deep C:",deep_xyz_c, deep_xyz_c_2)
    print("Peaks:", peaks)

    diagram_loop, diagrams = ph_diagrams
    ph_intersections = peaks[0]  # explude the peak range

    # select which tail
    if terminus in "Nn":
        deep_xyz = deep_xyz_n
        shallow_index = shallow_indices_n
        deep_index = deep_indices_n
        tail_xyz = tailN
    else:
        deep_xyz = deep_xyz_c
        shallow_index = shallow_indices_c
        deep_index = deep_indices_c
        tail_xyz = tailC


    """
        "pdb": lasso.pdb,
        "chain": lasso.chain,
        "bridge": lasso.bridge,
        "symbol": lasso.lassoprot_data["symbol"],
        "xyz": {"n":tailN, "loop": loop, "c": tailC},
        "shallow_n": shallow_index_n,
        "deep_n": deep_index_n,
        "deep_xyz_n": deep_xyz_n,
        "deep_str_n": deep_str_n,
        "shallow_c": shallow_index_c,
        "deep_c": deep_index_c,
        "deep_xyz_c": deep_xyz_c,
        "deep_str_c": deep_str_c,        
    """

    current_atom_index = 0

    # start plotting
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection='3d')  # 3d protein
    ax2 = fig.add_subplot(132)  # PH diagram
    ax3 = fig.add_subplot(133)  # minmax diagram

    # draw 3D protein
    scatter = plot_3D_lasso(tailN, loop, tailC, deep_xyz, terminus, current_atom_index, ax1)

    # draw persistence diagram
    plot_diagrams(diagrams[current_atom_index], show=False, ax=ax2)
    x_limits, y_limits = ax2.get_xlim(), ax2.get_ylim()  # we want to keep this constant during the animation

    # draw cost function
    scatter_cost = draw_cost_function(f=f_smooth, compl_f=f_bottle,
                                      indices=(shallow_index, deep_index, ph_intersections, current_atom_index), ax=ax3,
                                      for_paper=for_paper)

    # slider
    fig.subplots_adjust(bottom=0.25)  # make space for slider
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03])
    atom_slider = Slider(ax=ax_slider, label='Atom', valmin=0, valmax=len(tail_xyz) - 1, valinit=current_atom_index,
                         valstep=1)

    plt.legend(loc="upper right")
    # The function to be called anytime a slider's value changes
    def update(current_atom_index):

        # plot atom to 3D graph
        scatter._offsets3d = [(_,) for _ in tail_xyz[current_atom_index]]  # (x,y,z) -> ((x,),(y,),(z,))
        ax1.set_title(f"Atom {tail_xyz[current_atom_index][0]}{tail_xyz[current_atom_index][1]}")

        # draw PH diagram
        ax2.clear()
        plot_diagrams(diagrams[current_atom_index], show=False, ax=ax2)
        ax2.set_xlim(x_limits)
        ax2.set_ylim(y_limits)

        # draw point on cost/lifetime graph
        scatter_cost._offsets = [[current_atom_index, f_smooth[current_atom_index]]]
        # ax3.clear()
        # scatter_cost = draw_cost_function(f=y_dist_filtered, compl_f=y_dist,
        #                                   indices=(shallow_index, deep_index, ph_intersections, current_atom_index),
        #                                   ax=ax3)

        fig.canvas.draw_idle()

    # register the update function with slider
    atom_slider.on_changed(update)
    plt.show()



def OLD_interactive_ph_plot(lasso,
                        terminus,
                        threshold_mult_bottleneck=0.3,
                        filter_window=6,
                        threshold_mult_filtered=0.1,
                        for_paper=False):
    """
    Generates an interactive plot displaying:
    1. A 3D visualization of a protein lasso motif.
    2. A persistence homology (PH) diagram.
    3. A cost function graph derived from PH.

    The visualization includes a **slider** that allows users to interactively
    navigate through atoms in the tail of the protein and observe changes in
    their PH properties.

    Parameters
    ----------
    lasso : Lasso
        An instance of the Lasso class representing the lasso motif in the protein.
    terminus : str
        Either "N" or "C" specifying whether to analyze the N-terminus or C-terminus.
    threshold_mult_bottleneck : float, optional
        Multiplication factor for bottleneck distance thresholding (default: 0.3).
    filter_window : int, optional
        Window size for smoothing and filtering the cost function (default: 6).
    threshold_mult_filtered : float, optional
        Threshold multiplier for selecting relevant points in the cost function (default: 0.1).

    Returns
    -------
    None
        The function generates an interactive visualization but does not return any values.
    """

    # get loop coords & tail data
    tailN, loop, tailC = lasso.get_coords()
    tail_xyz, tail_atoms, shallow_index, deep_index, deep_xyz, deep_str = get_terminus_data(lasso, terminus)

    # compute persistence homologies by adding atoms from the tail to the loop
    diagrams = ph_lasso(lasso, terminus, dimension=1)  #ph_lasso_precomputed(loop, tail_xyz)
    diagram_loop = ph_lasso(lasso, None, dimension=1)[0]  #ph_lasso_precomputed(loop, tail_xyz)

    # compute the distance functions and intersections from PH

    y_dist = bottleneck_dist(diagram_loop=diagram_loop, diagrams=diagrams, threshold_mult=threshold_mult_bottleneck)

    if for_paper:
        print(list(y_dist))

    #y_dist_filtered = smooth_and_threshold(data=y_dist, window=filter_window, threshold_mult=threshold_mult_filtered)                 #4 and 0.05 at the beginning
    y_dist_filtered = smooth_and_threshold(data=y_dist, window=filter_window, threshold_mult=threshold_mult_filtered)                 #4 and 0.05 at the beginning
    ph_intersections = find_maxima_with_ranges(data=y_dist_filtered)[0]    #4 and 0.05 at the beginning

    if for_paper:
        print(list(y_dist_filtered))

    current_atom_index = 0

    # start plotting
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection='3d')  # 3d protein
    ax2 = fig.add_subplot(132)  # PH diagram
    ax3 = fig.add_subplot(133)  # minmax diagram

    # draw 3D protein
    scatter = plot_3D_lasso(tailN, loop, tailC, deep_xyz, terminus, current_atom_index, ax1)

    # draw persistence diagram
    plot_diagrams(diagrams[current_atom_index], show=False, ax=ax2)
    x_limits, y_limits = ax2.get_xlim(), ax2.get_ylim()  # we want to keep this constant during the animation

    # draw cost function
    scatter_cost = draw_cost_function(f=y_dist_filtered, compl_f=y_dist,
                                      indices=(shallow_index, deep_index, ph_intersections, current_atom_index), ax=ax3,
                                      for_paper=for_paper)

    # slider
    fig.subplots_adjust(bottom=0.25)  # make space for slider
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03])
    atom_slider = Slider(ax=ax_slider,label='Atom',valmin=0,valmax=len(tail_xyz)-1,valinit=current_atom_index,valstep=1)

    # The function to be called anytime a slider's value changes
    def update(current_atom_index):

        # plot atom to 3D graph
        scatter._offsets3d = [(_,) for _ in tail_xyz[current_atom_index]]  # (x,y,z) -> ((x,),(y,),(z,))
        ax1.set_title(f"Atom {tail_atoms[current_atom_index][0]}{tail_atoms[current_atom_index][1]}")

        # draw PH diagram
        ax2.clear()
        plot_diagrams(diagrams[current_atom_index], show=False, ax=ax2)
        ax2.set_xlim(x_limits)
        ax2.set_ylim(y_limits)

        # draw point on cost/lifetime graph
        scatter_cost._offsets = [[current_atom_index, y_dist_filtered[current_atom_index]]]
        # ax3.clear()
        # scatter_cost = draw_cost_function(f=y_dist_filtered, compl_f=y_dist,
        #                                   indices=(shallow_index, deep_index, ph_intersections, current_atom_index),
        #                                   ax=ax3)

        fig.canvas.draw_idle()

    # register the update function with slider
    atom_slider.on_changed(update)
    plt.show()