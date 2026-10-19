"""Official aic_model policy entry point."""
from .configuration import apply_configuration
apply_configuration()
from .controller import FinalInsertionPolicy as _Controller

class FinalInsertionPolicy(_Controller):
    def __init__(self, parent_node):
        apply_configuration()
        super().__init__(parent_node)
