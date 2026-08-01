"""Exit 0 only if a CUDA tensor can actually be allocated (real readiness test,
not just torch.cuda.is_available()). Used by the auto-resume watcher."""
import sys
try:
    import torch
    x = torch.zeros(1).cuda()
    _ = (x + 1).cpu()
    sys.exit(0)
except Exception as e:
    print(f"gpu not ready: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
