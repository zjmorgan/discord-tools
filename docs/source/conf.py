# Configuration file for the Sphinx documentation builder.
# Adapted for the `discord` package from a NeuXtalViz example.

import os
import sys
import inspect
import importlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

project = "discord"
author = "Zachary Morgan"

# -- General configuration ---------------------------------------------------

# Sphinx extensions. Keep these minimal but useful; add more as needed.
extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

root_doc = "index"

autosummary_generate = True
autodoc_typehints = "both"
add_module_names = False

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_permalinks_icon = "#"
html_show_sourcelink = False
html_copy_source = True

# -- Extension configuration -------------------------------------------------

plot_pre_code = """
import numpy as np
np.random.seed(13)

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.ioff()

"""

plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False
plot_basedir = ""


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object definitions on GitHub.

    This builds a link to the lines in the repository `zjmorgan/discord` on
    the `main` branch. If source lines can't be found, a fallback URL to the
    module file is returned.
    """
    if domain != "py":
        return None
    if not info.get("module"):
        return None

    baseurl = "https://github.com/zjmorgan/discord/blob/main/src/{}.py"
    filename = info["module"].replace(".", "/")
    url = baseurl.format(filename)

    try:
        mod = importlib.import_module(info["module"])
    except Exception:
        return url

    # Try to find the object and its source lines
    try:
        objname, *attr = info.get("fullname", "").split(".")
        obj = getattr(mod, objname) if objname else mod
        for a in attr:
            obj = getattr(obj, a)
        lines, start = inspect.getsourcelines(obj)
        stop = start + len(lines) - 1
        return f"{url}#L{start}-L{stop}"
    except (OSError, IOError, TypeError, AttributeError, IndexError):
        return url


def skip_qt_members(app, what, name, obj, skip, options):
    try:
        from PyQt5.QtCore import pyqtSignal, QMetaObject
    except Exception:
        try:
            from qtpy.QtCore import Signal as pyqtSignal, QMetaObject
        except Exception:
            pyqtSignal = None
            QMetaObject = None
    if pyqtSignal is not None and isinstance(obj, pyqtSignal):
        return True
    if QMetaObject is not None and isinstance(obj, QMetaObject):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_qt_members)
