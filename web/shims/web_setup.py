"""One-time setup of the browser version, imported by streamlit_app.py (so it runs once, not on every rerun)."""
import base64
import mimetypes
import re
from pathlib import Path
from streamlit.components.lib.local_component_registry import LocalComponentRegistry
from streamlit.runtime import get_instance

DASHBOARD = Path(__file__).parent / 'dianna' / 'dashboard'

# Custom components register with the Streamlit runtime that is running when their module is imported. All apps
# share one Python (index.html's shared worker, for remount()), so a later app gets a new runtime but reuses the
# imported modules: remember the components, so that register_components() can register them with every runtime.
_components = {}
_register_component = LocalComponentRegistry.register_component


def _remember_and_register(self, component):
    if component.name not in _components and component.abspath:
        _embed_stylesheet_urls(Path(component.abspath))
    _components[component.name] = component
    _register_component(self, component)


def _embed_stylesheet_urls(directory):
    """Embed the files that the stylesheets of a component's frontend refer to (e.g. fonts) as data: URIs.

    stlite inlines a component's stylesheets into its iframe, after which their relative url()s would resolve against
    the page instead of the stylesheet, e.g. to outside of the site when it is not at the root of its domain.
    """

    def data_uri(match, css):
        target = (css.parent / re.split('[?#]', match[2])[0]).resolve()
        if not target.is_file():
            return match[0]
        mime = mimetypes.guess_type(
            target.name)[0] or 'application/octet-stream'
        return f'url(data:{mime};base64,{base64.b64encode(target.read_bytes()).decode()})'

    for css in directory.rglob('*.css'):
        text = css.read_text()
        embedded = re.sub(r'url\((["\']?)(?![a-z]+:|/|#)([^"\')]+)\1\)',
                          lambda m, css=css: data_uri(m, css),
                          text)
        if embedded != text:
            css.write_text(embedded)


LocalComponentRegistry.register_component = _remember_and_register


def register_components():
    """Register the custom components imported so far with the runtime of the current app."""
    registry = get_instance().component_registry
    for component in _components.values():
        if registry.get_component_path(component.name) is None:
            registry.register_component(component)


# stlite (1.12) does not support Streamlit's Custom Components v2, which streamlit-drawable-canvas 0.13 uses.
# The browser installs 0.9.3 instead, which always returns the image data and lacks the `return_image_data` argument.
import streamlit_drawable_canvas  # noqa: E402 (after the patch above, so that its component is remembered)

_st_canvas = streamlit_drawable_canvas.st_canvas
streamlit_drawable_canvas.st_canvas = lambda *args, return_image_data=True, **kwargs: _st_canvas(
    *args, **kwargs)
