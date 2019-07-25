import matplotlib

from matplotlib.backends.backend_pgf import FigureCanvasPgf


SIGCONF_RCPARAMS = {
    "figure.figsize": (7.0, 3.0),       # Column width: 3.333 in, space between cols: 0.333 in.
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Linux Libertine O",  # use "Linux Libertine" as the standard font
    "font.size": 9,
    "axes.titlesize": 9,                # LaTeX default is 10pt font.
    "axes.labelsize": 7,                # LaTeX default is 10pt font.
    "legend.fontsize": 7,               # Make the legend/label fonts a little smaller
    "legend.frameon": False,            # Remove the black frame around the legend
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\setmainfont{Linux Libertine O}',
        r'\setmathfont{Linux Libertine O}',
    ]
}


def sigconf_settings():
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(SIGCONF_RCPARAMS)
    print("Sigconf settings loaded!")
