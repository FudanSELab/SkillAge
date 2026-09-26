class SkillAgeError(Exception):
    """Base exception for the package."""


class BundleValidationError(SkillAgeError):
    """Raised when a dataset bundle is incomplete or has been modified."""


class ConfigurationError(SkillAgeError):
    """Raised when a requested dataset or training configuration is invalid."""


class BudgetInvariantError(SkillAgeError):
    """Raised when replay-certified degradation exceeds the available budget."""

