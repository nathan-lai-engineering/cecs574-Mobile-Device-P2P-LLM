"""
Patch the installed transformers LLaMA implementation for:
1. torch.fx symbolic tracing compatibility (remove dynamic control-flow checks)
2. GQA (Grouped Query Attention) support in transformers 4.33.x
   - TinyLlama uses 4 KV heads vs 32 query heads; transformers 4.33.x
     uses self.num_key_value_heads which is not always set, and the k/v
     projection shapes may reflect the wrong head count at init time.
     Fix: use self.k_proj.weight.shape[0] // self.head_dim (actual loaded weights).
"""
import re
import shutil
import sys

import transformers.models.llama.modeling_llama as _m

path = _m.__file__
shutil.copy(path, path + '.bak')
print(f"Backed up to {path}.bak")

with open(path) as f:
    src = f.read()

original = src

# ---------------------------------------------------------------------------
# Fix 1: LlamaRotaryEmbedding.forward — remove seq_len cache-refresh guard
# The `if seq_len > self.max_seq_len_cached:` block compares a proxy to an int
# which breaks torch.fx tracing.
# ---------------------------------------------------------------------------
src = re.sub(
    r'[ \t]+if seq_len > self\.max_seq_len_cached:.*?self\.sin_cached = self\.sin_cached\.to\(.*?\)\n',
    '',
    src,
    flags=re.DOTALL,
)

# ---------------------------------------------------------------------------
# Fix 2: LlamaAttention.forward — remove tensor-shape validation checks
# These compare tensor shapes (proxies during tracing) to integers.
# ---------------------------------------------------------------------------
# Pattern: `if attn_weights.size() != (...): raise ValueError(...)`
src = re.sub(
    r'[ \t]+if attn_weights\.size\(\) != \(.*?\):\n[ \t]+raise ValueError\(\n.*?\)\n',
    '',
    src,
    flags=re.DOTALL,
)
# Pattern: `if attention_mask.size() != (...): raise ValueError(...)`
src = re.sub(
    r'[ \t]+if attention_mask\.size\(\) != \(.*?\):\n[ \t]+raise ValueError\(\n.*?\)\n',
    '',
    src,
    flags=re.DOTALL,
)

# ---------------------------------------------------------------------------
# Fix 3: GQA — use k_proj.weight.shape[0] for num_key_value_heads
# transformers 4.33.x uses self.num_key_value_heads, but this attribute may
# not exist, or at init time the projection may have been sized for num_heads.
# Using weight.shape[0] reads the actually-loaded tensor dimensions.
# ---------------------------------------------------------------------------
src = src.replace(
    'key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)',
    'key_states = key_states.view(bsz, q_len, self.k_proj.weight.shape[0] // self.head_dim, self.head_dim)',
)
src = src.replace(
    'value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)',
    'value_states = value_states.view(bsz, q_len, self.v_proj.weight.shape[0] // self.head_dim, self.head_dim)',
)

# Also patch num_key_value_groups calculation if it uses self.num_key_value_heads directly
src = src.replace(
    'self.num_key_value_groups = self.num_heads // config.num_key_value_heads',
    'self.num_key_value_groups = 1  # patched — computed dynamically in forward',
)
# num_kv_groups used inline in forward:
src = src.replace(
    'key_states = repeat_kv(key_states, self.num_key_value_groups)',
    'key_states = repeat_kv(key_states, self.num_heads // (self.k_proj.weight.shape[0] // self.head_dim))',
)
src = src.replace(
    'value_states = repeat_kv(value_states, self.num_key_value_groups)',
    'value_states = repeat_kv(value_states, self.num_heads // (self.v_proj.weight.shape[0] // self.head_dim))',
)

# ---------------------------------------------------------------------------
# Fix 4: Add repeat_kv if missing (needed for GQA, absent in some 4.33.x builds)
# ---------------------------------------------------------------------------
if 'def repeat_kv' not in src:
    repeat_kv_code = '''
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads n_rep times to match query heads (GQA expansion)."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        .reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    )

'''
    # Insert before the first class definition
    src = re.sub(r'(^class Llama)', repeat_kv_code + r'\1', src, count=1, flags=re.MULTILINE)

if src == original:
    print("WARNING: no changes were made — check patterns match the installed file version")
else:
    with open(path, 'w') as f:
        f.write(src)
    print(f"Patched successfully: {path}")

# Verify the key replacements landed
checks = [
    ('k_proj.weight.shape[0] // self.head_dim', 'GQA fix for key_states'),
    ('v_proj.weight.shape[0] // self.head_dim', 'GQA fix for value_states'),
]
with open(path) as f:
    patched = f.read()

all_ok = True
for snippet, label in checks:
    if snippet in patched:
        print(f"  [OK] {label}")
    else:
        print(f"  [MISSING] {label}")
        all_ok = False

if not all_ok:
    print("Some patches did not apply. Check modeling_llama.py manually.")
    sys.exit(1)
