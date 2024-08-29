import nbformat
from nbconvert import HTMLExporter

custom_css = """
<style>
body { background-color: #F0F0F0 !important; }
[data-mime-type='application/vnd.jupyter.stderr'] { display: none; }
</style>
"""

files = {
    "workbench/TabularModel.ipynb": "output/TabularModel.html",
    "workbench/EDA.ipynb": "output/EDA.html",
}


def export_html(exclude_input, html_output_path):
    exporter = HTMLExporter()
    exporter.exclude_input = exclude_input

    html, _ = exporter.from_notebook_node(nb)

    if not exclude_input:
        html_output_path = html_output_path.replace(".html", "_with_code.html")

    return html, html_output_path


for notebook_path, html_output_path in files.items():
    for exclude_input in [True, False]:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        html, html_output_path = export_html(exclude_input, str(html_output_path))
        html = html.replace("</head>", f"{custom_css}</head>")

        with open(html_output_path, "w") as f:
            f.write(html)
