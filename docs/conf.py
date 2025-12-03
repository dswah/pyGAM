# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
autoapi_generate_api_docs = False
autoapi_dirs = ["../"]

project = "pyGAM"
copyright = "2025 pyGAM Developers"
author = "pyGAM Developers  "
index_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "numpydoc",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "search_bar_text": "Search",
    "footer_start": ["copyright"],
    "footer_end": ["footer"],
    "logo": {
        "text": "pyGAM",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pygam",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/dswah/pyGAM",
            "icon": "fa-brands fa-github",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "use_edit_page_button": True,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 2,
}

# Remove left side bar from following pages
html_sidebars = {
    "notebooks/*": [],
    "Quick Start": [],
}
html_context = {
    "github_user": "dswah",
    "github_repo": "pyGAM",
    "github_version": "main",
    "doc_path": "docs",
    "contributing": "https://github.com/dswah/pyGAM/issues",
}

html_static_path = ["_static"]

html_logo = "../imgs/pygam_tensor.png"
html_favicon = "../imgs/pygam_tensor.png"
htmlhelp_basename = "pyGAM Docs"


# -- Options for AutoAPI ----------------------------------------------------
autoapi_python_class_content = "class"

# -- Autosummary ------------------------------------------------------------
autosummary_generate = True

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        index_doc,
        "pyGAM.tex",
        "pyGAM Documentation",
        "pyGAM Developers",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(index_doc, "pygam", "pyGAM Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        index_doc,
        "pyGAM",
        "pyGAM Documentation",
        author,
        "pyGAM",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


def setup_to_main(
    app: Sphinx, pagename: str, templatename: str, context, doctree
) -> None:
    """
    Add a function that jinja can access for returning an "edit this page" link
    pointing to `main`.
    """

    def to_main(link: str) -> str:
        """
        Transform "edit on github" links and make sure they always point to the
        main branch.

        Args:
            link: the link to the github edit interface

        Returns
        -------
            the link to the tip of the main branch for the same file
        """
        links = link.split("/")
        idx = links.index("edit")
        return "/".join(links[: idx + 1]) + "/main/" + "/".join(links[idx + 2 :])

    context["to_main"] = to_main


def setup(app: Sphinx) -> Dict[str, Any]:
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application
    Returns:
        the 2 parallel parameters set to ``True``.
    """
    app.connect("html-page-context", setup_to_main)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
