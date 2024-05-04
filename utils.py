import nbformat
from nbconvert import PythonExporter
import io
import contextlib
import traceback
from textwrap import wrap


# def read_normalized_notebook(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         nb = nbformat.read(f, as_version=4)
#     nbformat.validate(nb)  # Optional: validate the notebook first
#     nb = nbformat.normalize(nb)  # Normalize the notebook


def has_string_in_cell(notebook_path, cell_index, search_string, case_sensitive=False):
    """
    Checks if a specific string or sentence exists within the specified cell of a Jupyter notebook.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param cell_index: Index of the cell to check.
    :param search_string: The string to search for in the cell.
    :param case_sensitive: Boolean to indicate if the search should be case sensitive.
    :return: Boolean indicating if the string is found in the specified cell.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Ensure the cell index is within the range of existing cells
    if cell_index >= len(nb.cells):
        raise IndexError("The provided cell index is out of range for this notebook.")
    
    cell = nb.cells[cell_index]
    cell_content = cell.source
    
    # Adjust string matching based on case sensitivity
    if not case_sensitive:
        cell_content = cell_content.lower()
        search_string = search_string.lower()
    
    return search_string in cell_content

def has_string_in_code_cells(notebook_path, start_index, end_index, search_string, case_sensitive=False):
    """
    Checks if a specific string or sentence exists within code cells in a specified index range of a Jupyter notebook.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param start_index: Starting index of the cell range to check (inclusive).
    :param end_index: Ending index of the cell range to check (exclusive).
    :param search_string: The string to search for in the cells.
    :param case_sensitive: Boolean to indicate if the search should be case sensitive.
    :return: Boolean indicating if the string is found within any of the specified code cells.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Adjust string matching based on case sensitivity
    if not case_sensitive:
        search_string = search_string.lower()
    
    for cell_index in range(start_index, min(end_index, len(nb.cells))):
        cell = nb.cells[cell_index]
        if cell.cell_type == 'code':  # Only check code cells
            cell_content = cell.source
            if not case_sensitive:
                cell_content = cell_content.lower()

            if search_string in cell_content:
                return True
    
    return False

def print_text_and_output_cells(notebook_path, start_index, end_index, line_width=80):
    """
    Prints the content of text (markdown) and output cells in a specified index range of a Jupyter notebook,
    wrapping text lines to a specified width for better readability.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param start_index: Starting index of the cell range to print from (inclusive).
    :param end_index: Ending index of the cell range to print to (exclusive).
    :param line_width: Maximum width of the text lines, default is 80 characters.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for cell_index in range(start_index, min(end_index, len(nb.cells))):
        cell = nb.cells[cell_index]
        
        if cell.cell_type == 'markdown':
            print(f"Markdown Cell {cell_index}:")
            wrapped_text = "\n".join(wrap(cell.source, width=line_width))
            print(wrapped_text)
            print("-" * 40)  # Separator for readability
        
        elif cell.cell_type == 'code':
            print(f"Code Cell {cell_index}:")
            for output in cell.outputs:
                if output.output_type == 'stream':
                    wrapped_text = "\n".join(wrap(output.text, width=line_width))
                    print(wrapped_text)
                elif output.output_type == 'execute_result' or output.output_type == 'display_data':
                    if 'text/plain' in output.data:
                        wrapped_text = "\n".join(wrap(output.data['text/plain'], width=line_width))
                        print(wrapped_text)
                    if 'image/png' in output.data:
                        print("<Image output not shown>")
                elif output.output_type == 'error':
                    wrapped_text = "\n".join(wrap(f"Error: {output.ename}, {output.evalue}", width=line_width))
                    print(wrapped_text)
            print("-" * 40)  # Separator for readability

def print_code_and_output_cells(notebook_path, start_index, end_index, line_width=80):
    """
    Prints the content of code and output cells in a specified index range of a Jupyter notebook,
    wrapping text lines to a specified width for better readability.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param start_index: Starting index of the cell range to print from (inclusive).
    :param end_index: Ending index of the cell range to print to (exclusive).
    :param line_width: Maximum width of the text lines, default is 80 characters.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for cell_index in range(start_index, min(end_index, len(nb.cells))):
        cell = nb.cells[cell_index]
        
        if cell.cell_type == 'code':
            print(f"Code Cell {cell_index}:")
            for output in cell.outputs:
                if output.output_type == 'stream':
                    wrapped_text = "\n".join(wrap(output.text, width=line_width))
                    print(wrapped_text)
                elif output.output_type == 'execute_result' or output.output_type == 'display_data':
                    if 'text/plain' in output.data:
                        wrapped_text = "\n".join(wrap(output.data['text/plain'], width=line_width))
                        print(wrapped_text)
                    if 'image/png' in output.data:
                        print("<Image output not shown>")
                elif output.output_type == 'error':
                    wrapped_text = "\n".join(wrap(f"Error: {output.ename}, {output.evalue}", width=line_width))
                    print(wrapped_text)
            print("-" * 40)  # Separator for readability


def find_cells_by_indices(notebook_path, indices):
    """
    Finds cells by their indices in a Jupyter notebook and returns their details.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param indices: List of integers representing the indices of cells to find.
    :return: List of dictionaries, each containing the index, cell_id, cell_type, and content of the specified cells.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found_cells = []
    
    for index in indices:
        if index < len(nb.cells):
            cell = nb.cells[index]
            cell_info = {
                "index": index,
                "cell_id": cell.get("id", "N/A"),  # Handle the possibility of missing 'id' field
                "cell_type": cell.cell_type,
                "content": cell.source  # Include the content of the cell
            }
            found_cells.append(cell_info)
        else:
            print(f"Warning: No cell at index {index}. Skipping...")
    
    return found_cells

def find_cells_with_text(notebook_path, search_text, case_sensitive=False):
    """
    Searches for cells containing specified text in a Jupyter notebook and returns their contents.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param search_text: Text to search for within the notebook cells.
    :param case_sensitive: Boolean to indicate if the search should be case sensitive.
    :return: List of dictionaries, each containing the index, cell_id, cell_type, and content of cells where the text is found.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found_cells = []
    
    # Normalize text based on case sensitivity
    if not case_sensitive:
        search_text = search_text.lower()
    
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' or cell.cell_type == 'markdown':
            # Retrieve cell content and normalize if case insensitive
            cell_content = cell.source
            if not case_sensitive:
                cell_content = cell_content.lower()
            
            # Check if search_text is in cell_content
            if search_text in cell_content:
                cell_info = {
                    "index": index,
                    "cell_id": cell.get("id", "N/A"),  # Some cells might not have an 'id' field
                    "cell_type": cell.cell_type,
                    "content": cell.source  # Return the original (non-normalized) content
                }
                found_cells.append(cell_info)
    
    return found_cells

def execute_cell(python_code, namespace):
    """Execute Python code safely and capture stdout."""
    try:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(python_code, namespace)
    except Exception as e:
        # If you want to log the error to stdout or another logging system
        print(f"Error executing the code: {traceback.format_exc()}")

def extract_variables(notebook_path, cell_idx=-1):
    """
    Extracts variables from the student's notebook up to the specified cell index.
    :param notebook_path: Path to the notebook file.
    :param cell_idx: Index of the last cell to execute, default -1 (execute all cells).
    :return: Dictionary of the final state of all variables up to the specified cell.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    namespace = {}

    # Iterate over the cells, execute them up to the specified cell index or all if cell_idx is -1
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if cell_idx != -1 and index > cell_idx:
                break  # Stop processing if we reach the specified cell index
            python_code = exporter.from_notebook_node(nbformat.v4.new_notebook(cells=[cell]))[0]
            execute_cell(python_code, namespace)
    
    return namespace


def extract_initial_variables(notebook_path):
    """Extracts all variables right after they are loaded in the notebook for the first time."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    namespaces = {}

    cell_namespace = {}

    for cell in nb.cells:
        if cell.cell_type == 'code':
            python_code = exporter.from_notebook_node(nbformat.v4.new_notebook(cells=[cell]))[0]

            execute_cell(python_code, cell_namespace)
            
            # Update namespace with new variables found in this cell, only if they
            # were not previously set
            for key, _ in cell_namespace.items():
                if key not in namespaces:
                    namespaces[key] = cell_namespace[key]
    
    return namespaces
