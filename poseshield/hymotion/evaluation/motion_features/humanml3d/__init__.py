"""Vendored HumanML3D motion-feature conversion (used only by the MFD metric).

These four files are copied from the HumanML3D / momask-codes reference
implementation (EricGuo5513) and are what turn raw joint positions into the
263-D "redundant representation" whose first 67 channels the MFD motion encoder
expects. They are vendored rather than reimplemented so the features fed to the
pretrained encoder match the ones it was trained on exactly.

Modifications to the upstream source are marked with an inline `# PATCH`
comment. They are limited to:

- numpy>=1.24 / scipy>=1.15 compatibility (`np.float` and
  `scipy.ndimage.filters` were removed upstream).
- Rewriting the original top-level `from common...` / `from utils...` imports as
  relative imports, so this package does not need to be injected onto
  `sys.path` and cannot collide with any other module named `common` or `utils`.

No numerical behavior is changed by these patches.
"""
