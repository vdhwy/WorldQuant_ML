"""
Core generation components

The heavy submodules use package-relative imports (`from ..ollama import ...`)
and can therefore only be re-exported when this package is imported as part
of the parent `alpha_mining` package. When the repo root itself is on
sys.path (flat layout, e.g. scripts doing `from core.brain_api import poll`),
those re-exports would explode with "attempted relative import beyond
top-level package" -- so they are guarded and become optional.
"""

from .simulator_tester import SimulatorTester, SimulationSettings, SimulationResult

try:  # requires parent-package context (alpha_mining.*)
    from .template_generator import TemplateGenerator
except ImportError:  # pragma: no cover - flat-layout mode
    TemplateGenerator = None

try:  # requires parent-package context (alpha_mining.*)
    from .enhanced_template_generator_v3 import EnhancedTemplateGeneratorV3
except ImportError:  # pragma: no cover - flat-layout mode
    EnhancedTemplateGeneratorV3 = None

__all__ = [
    'SimulatorTester',
    'SimulationSettings',
    'SimulationResult',
    'TemplateGenerator',
    'EnhancedTemplateGeneratorV3'
]
