# Local workflow privacy review

The local workflow contained account-specific workspace and archive paths, device/domain selections and operational setup details. No credentials were found in that file. The current and historical workflow files are retained locally, excluded from tracking and ignored by Git. Public links point to the generic [workflow](../WORKFLOW.md).

This change removes the files from the current repository tree, not earlier Git history. Other historical project records may still contain machine-specific paths; this was a focused workflow review, not a repository-wide privacy audit.
