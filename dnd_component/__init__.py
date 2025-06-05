import streamlit.components.v1 as components
import os

# Set to False for development (if you're running a local dev server for the frontend)
# Set to True for production (when you have a static build of the frontend)
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "dnd_visualizer",
        url="http://localhost:3001",  # URL of the frontend development server
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes a 'build' subfolder
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "dnd_visualizer", path=build_dir)


def dnd_visualizer_component(columns, key=None):
    """
    Streamlit component for drag-and-drop visualization configuration.

    Parameters:
    -----------
    columns : list
        A list of column names to be made draggable.
    key : str, optional
        Streamlit key for the component.

    Returns:
    --------
    dict
        A dictionary representing the configuration from the drop zones,
        e.g., {'x_axis': 'col_A', 'y_axes': ['col_B', 'col_D'], 'color': 'col_C'}
        Returns None or a default dictionary if no interaction yet.
    """
    default_value = {'x_axis': None, 'y_axes': [], 'color': None,
                     'size': None, 'facet_row': None, 'facet_col': None}
    component_value = _component_func(
        columns=columns, key=key, default=default_value)
    return component_value
