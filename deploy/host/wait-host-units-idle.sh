#!/usr/bin/env bash
# Holds the deploy converge while host-checkout kca units (no kca-run wrapper) are still running.
# Usage: wait-host-units-idle.sh <checkout_dir> <timeout_s> <poll_s>
# Exits 0 once every such unit is idle; exits 1 on timeout. Never stops or kills a unit.
set -euo pipefail

checkout=${1:?checkout dir required}
timeout_s=${2:?timeout seconds required}
poll_s=${3:?poll seconds required}

units=()
for unit_file in "$checkout"/deploy/systemd/kca-*.service; do
  [ -e "$unit_file" ] || continue
  name=$(basename "$unit_file" .service)
  [[ "$name" == *@* ]] && continue
  grep -q 'kca-run' "$unit_file" && continue
  units+=("$name")
done

busy=()
collect_busy() {
  local unit state
  busy=()
  for unit in "${units[@]}"; do
    state=$(systemctl --user show -p ActiveState --value "$unit")
    case "$state" in
      active | activating | deactivating | reloading) busy+=("$unit") ;;
    esac
  done
}

start=$SECONDS
while true; do
  collect_busy
  if [ "${#busy[@]}" -eq 0 ]; then
    echo "[SYS] stage=host_units_wait status=clear waited_s=$((SECONDS - start))"
    exit 0
  fi
  echo "[SYS] stage=host_units_wait busy=$(IFS=,; echo "${busy[*]}")"
  if [ $((SECONDS - start)) -ge "$timeout_s" ]; then
    echo "[SYS] stage=host_units_wait status=timeout busy=$(IFS=,; echo "${busy[*]}")" >&2
    exit 1
  fi
  sleep "$poll_s"
done
