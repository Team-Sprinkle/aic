#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim 5.1 blocks RTX rendering on Linux drivers below 535.129.03.
# Some shared/rootless AIC hosts cannot update the host driver from inside the
# user session, but cameras may still work after bypassing this guard. This
# patch is intentionally explicit and local to the container filesystem.

DRIVER_REQUIREMENTS="${DRIVER_REQUIREMENTS:-/isaac-sim/kit/driver-requirements.json}"
BACKUP="${DRIVER_REQUIREMENTS}.aic_backup"

if [[ "${AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to patch Isaac RTX driver requirements without explicit opt-in.

Set:
  AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1

Use this only after confirming the host driver cannot be updated and validating
camera tensor smoke immediately after patching.
EOF
  exit 2
fi

if [[ ! -f "${DRIVER_REQUIREMENTS}" ]]; then
  echo "Driver requirements file not found: ${DRIVER_REQUIREMENTS}" >&2
  exit 1
fi

if [[ ! -f "${BACKUP}" ]]; then
  cp "${DRIVER_REQUIREMENTS}" "${BACKUP}"
fi

perl -0pi -e '
  s/"message": "The minimum Omniverse RTX requirement on Linux",\n\s+"category": \[\],\n\s+"blocked": true,\n\s+"recommended": false,/"message": "The minimum Omniverse RTX requirement on Linux disabled for AIC rootless camera smoke",\n                "category": [],\n                "blocked": false,\n                "recommended": false,/s
' "${DRIVER_REQUIREMENTS}"

if ! grep -A4 "minimum Omniverse RTX requirement on Linux" "${DRIVER_REQUIREMENTS}" | grep -q '"blocked": false'; then
  echo "Failed to patch Linux RTX driver requirement block in ${DRIVER_REQUIREMENTS}" >&2
  exit 1
fi

echo "Patched ${DRIVER_REQUIREMENTS}"
echo "Backup: ${BACKUP}"
echo "Run camera tensor smoke before training."
