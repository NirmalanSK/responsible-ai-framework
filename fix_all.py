with open('pipeline_v15.py', 'r') as f:
    content = f.read()

# Fix 1: Make sure imports are correct
old_imports = """from scm_engine import (
    SCMEngine, Severity,
)
try:
    from scm_engine_v2 import (
        SCMEngineV2 as _SCMV2,
        CausalFindings,        # v2 CausalFindings has all required fields
    )
    _SCM_V2_AVAILABLE = True
except ImportError:
    from scm_engine import CausalFindings  # fallback to v1
    _SCM_V2_AVAILABLE = False
# ── v15c: Sparse Causal Activation Matrix (scm_engine_v2) ─────────
try:
    from scm_engine_v2 import activate_matrix, get_domain_multiplier, DOMAIN_RISK_MULTIPLIER
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False"""

new_imports = """from scm_engine_v2 import (
    SCMEngineV2 as _SCMV2,
    CausalFindings,
    activate_matrix,
    get_domain_multiplier,
    DOMAIN_RISK_MULTIPLIER,
)"""

content = content.replace(old_imports, new_imports)

# Fix 2: Replace the engine instantiation
content = content.replace(
    'self.engine = _SCMV2()  # Always use v2 (no fallback)',
    'self.engine = _SCMV2()'
)

with open('pipeline_v15.py', 'w') as f:
    f.write(content)

print("✅ Fixed pipeline imports!")

