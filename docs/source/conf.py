# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join("..", ".."))) 
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pymob"
copyright = "2024, Florian Schunck"
author = "Florian Schunck"
release = "0.6.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    'sphinx.ext.mathjax',
    "sphinx.ext.duration",
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # used for building numpy style documentation
    # "sphinx_immaterial", # only needed for sphinx_immaterial theme
]

# https://myst-parser.readthedocs.io/en/latest/configuration.html
myst_enable_extensions = [
    "tasklist",
    "dollarmath",
    "amsmath"
]

templates_path = ["_templates"]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# html_theme = "sphinx_immaterial"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_context = {
   # ...
   "default_mode": "light"
}

autosummary_generate = True
autosummary_imported_members = True
autosectionlabel_prefix_document = True