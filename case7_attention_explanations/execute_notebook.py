"""Execute top to bottom in a fresh IPython process, capture rich outputs, export HTML.

Avoids relying on a local notebook-server port. Equivalent code-cell order is preserved.
"""
from pathlib import Path
import os,time,json
import nbformat
from nbconvert import HTMLExporter
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
path=ROOT/'07_attention_explanation_audit.ipynb'
book=nbformat.read(path,as_version=4)
shell=InteractiveShell.instance()
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from matplotlib_inline.backend_inline import configure_inline_support
configure_inline_support(shell,'inline')
start=time.time();count=0
for cell in book.cells:
    if cell.cell_type!='code':continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=True)
    cell.execution_count=count;cell.outputs=[]
    if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for rich in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    if result.error_before_exec or result.error_in_exec:
        nbformat.write(book,path)
        raise RuntimeError(f'Cell {count} failed: {result.error_before_exec or result.error_in_exec}')
    print('Executed cell',count,flush=True)
nbformat.validate(book);nbformat.write(book,path)
exporter=HTMLExporter();exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
html,_=exporter.from_notebook_node(book)
path.with_suffix('.html').write_text(html)
record={'code_cells':count,'errors':0,'seconds':time.time()-start,'execution':'fresh sequential IPython process; captured rich outputs'}
(ROOT/'outputs/notebook_execution.json').write_text(json.dumps(record,indent=2))
print(record)
