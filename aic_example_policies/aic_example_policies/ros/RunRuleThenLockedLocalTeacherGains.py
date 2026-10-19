"""Keep acquisition gains; use recorded teacher gains for insertion only."""
from .RunRuleThenLockedLocal import RunRuleThenLockedLocal


class RunRuleThenLockedLocalTeacherGains(RunRuleThenLockedLocal):
    local_translation_stiffness=90.
