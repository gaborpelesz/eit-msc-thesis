#!/usr/bin/env bash
# Everything the campaign needs from the host that needs root, in one place.
#
# The measurement environment of R-ENV-02 -- persistence mode on, clocks locked,
# no display on the benchmark GPU -- is host state, not harness state, and it is
# lost on a reboot or a driver reload. `bench run` re-checks it before every run
# and refuses when it is not there, except for the display: this driver reports
# `display_active: Disabled` even with a Wayland session live on the card, so
# that one is on us. Hence `headless`.
#
#   tools/host_prep.sh status          what the campaign needs vs what is here
#   sudo tools/host_prep.sh dirs       .data tree, owned by the invoking user
#   sudo tools/host_prep.sh hold       pin the NVIDIA packages for the campaign
#   sudo tools/host_prep.sh gpu <MHz>  persistence mode + locked clocks
#   sudo tools/host_prep.sh headless   stop the desktop  <- from a TTY or ssh
#   sudo tools/host_prep.sh restore    undo all of the above
#
# GPU=<index> selects the card (default 0).
set -euo pipefail

gpu=${GPU:-0}
owner=${SUDO_USER:-$(id -un)}
# All benchmark data lives under the repository's git-ignored .data/.
data_root=$(cd "$(dirname "$0")/../.." && pwd)/.data
dirs=("$data_root/bench/results" "$data_root/bench/work" "$data_root/bench/clouds" "$data_root/bench/diag" "$data_root/eth3d")

need_root() { [ "$(id -u)" -eq 0 ] || { echo "run this with sudo: sudo $0 $*" >&2; exit 1; }; }
nvidia_packages() { dpkg-query -W -f '${binary:Package} ${db:Status-Status}\n' 'nvidia-*' 2>/dev/null |
                    awk '$2 == "installed" {print $1}'; }

case "${1:-status}" in

dirs)
  need_root dirs
  mkdir -p "${dirs[@]}"
  chown "$owner:$owner" "${dirs[@]}"
  # A campaign outlives the login session that starts it, and `headless` below
  # tears that session down. Lingering keeps the user manager -- and so the tmux
  # the campaign runs in -- alive across it.
  loginctl enable-linger "$owner"
  echo "ready, owned by $owner, lingering enabled:"; printf '  %s\n' "${dirs[@]}"
  ;;

hold)
  # unattended-upgrades is enabled here and the NVIDIA packages come from an
  # allowed origin, so the driver can be replaced under a running campaign. The
  # userspace libraries would then no longer match the loaded kernel module and
  # every container would fail on it; a reload or reboot changes
  # `driver_version`, which is a load-bearing fingerprint field, and R-ENV-06
  # ends the campaign. Neither is something to discover on day three.
  need_root hold
  mapfile -t packages < <(nvidia_packages)
  [ "${#packages[@]}" -gt 0 ] || { echo "no NVIDIA packages found to hold" >&2; exit 1; }
  apt-mark hold "${packages[@]}"
  ;;

gpu)
  need_root gpu "${2:-<MHz>}"
  mhz=${2:-}
  [ -n "$mhz" ] || { echo "usage: sudo $0 gpu <MHz>   (from tools/clock_probe.sh)" >&2; exit 1; }
  nvidia-smi -i "$gpu" -pm 1
  nvidia-smi -i "$gpu" -lgc "$mhz,$mhz"
  echo
  nvidia-smi -i "$gpu" --query-gpu=index,name,uuid,persistence_mode,clocks.sm --format=csv
  echo "verify with: uv run bench fingerprint --image mvs-bench:sm120 --gpu $gpu"
  ;;

headless)
  # This kills the graphical session and every terminal inside it -- including
  # the one you are reading this in, if it is a desktop terminal.
  need_root headless
  # sudo does not pass XDG_SESSION_ID through, so fall back to the session the
  # calling shell sits in. A console or ssh session is Type=tty and survives;
  # anything else, including a session this cannot identify, is refused.
  session_id=${XDG_SESSION_ID:-$(grep -o 'session-[0-9]*\.scope' "/proc/$PPID/cgroup" 2>/dev/null |
                                 head -1 | tr -dc '0-9')}
  session_type=$(loginctl show-session "${session_id:-x}" -p Type --value 2>/dev/null || true)
  if [ "${FORCE:-0}" != "1" ] && [ "$session_type" != "tty" ]; then
    echo "this shell is in a '${session_type:-unidentified}' session; isolating would kill it." >&2
    echo "switch to a text console (ctrl+alt+F3) or ssh in, and run it there." >&2
    echo "to override anyway: FORCE=1 sudo $0 headless" >&2
    exit 1
  fi
  systemctl isolate multi-user.target
  echo "desktop stopped. The default target is unchanged, so a reboot brings it back"
  echo "-- and a reboot also drops the clock lock, so re-run 'gpu <MHz>' after one."
  ;;

restore)
  need_root restore
  nvidia-smi -i "$gpu" -rgc || true
  nvidia-smi -i "$gpu" -pm 0 || true
  mapfile -t packages < <(nvidia_packages)
  [ "${#packages[@]}" -eq 0 ] || apt-mark unhold "${packages[@]}" || true
  systemctl isolate graphical.target || true
  echo "clocks reset, persistence off, packages unheld, desktop back."
  ;;

status)
  echo "== directories"
  for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then printf '  ok      %-22s owner %s\n' "$dir" "$(stat -c %U "$dir")"
    else printf '  MISSING %s\n' "$dir"; fi
  done
  echo "== dataset (campaign-01's seven scenes x 2 widths)"
  for scene in delivery_area kicker meadow office pipes relief relief_2; do
    for width in 1600 3200; do
      dir="$data_root/eth3d/${scene}_${width}"
      if [ -d "$dir" ]; then printf '  ok      %s\n' "$dir"; else printf '  MISSING %s\n' "$dir"; fi
    done
  done
  echo "== image"
  image_id=$(docker image inspect mvs-bench:sm120 --format '{{.Id}}' 2>/dev/null || true)
  if [ -n "$image_id" ]; then echo "  ok      mvs-bench:sm120 $image_id"
  else echo "  MISSING mvs-bench:sm120"; fi
  echo "== NVIDIA packages"
  apt-mark showhold 2>/dev/null | sed 's/^/  held    /' | grep . || echo "  none held (see: $0 hold)"
  echo "== GPU $gpu"
  nvidia-smi -i "$gpu" --query-gpu=index,name,uuid,persistence_mode,clocks.sm,display_active \
    --format=csv | sed 's/^/  /'
  echo "== graphical session"
  systemctl is-active graphical.target >/dev/null 2>&1 &&
    echo "  RUNNING -- the desktop is up (see: $0 headless)" || echo "  ok      stopped"
  pgrep -a gnome-shell >/dev/null 2>&1 && echo "  gnome-shell is alive" || true
  ;;

*)
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
  ;;
esac
