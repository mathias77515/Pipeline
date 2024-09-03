# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os


sys.path.insert(0, os.path.abspath('../..'))

project = 'Pipeline'
copyright = '2024, M. Regnier, T. Laclavère'
author = 'M. Regnier, T. Laclavère'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

myst_enable_extensions = [
    "amsmath",            # Prise en charge des équations LaTeX
    "dollarmath",         # Support de l'inline math avec $
    "deflist",            # Support des listes de définition
    "html_admonition",    # Support des admonitions HTML (boîtes d'avertissement, etc.)
    "html_image",         # Support des balises d'images HTML
    "linkify",            # Convertit les URLs en liens cliquables
    "substitution",       # Support des substitutions de texte
    "tasklist",           # Support des listes de tâches
]

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
