#!/usr/bin/env bash
set -euo pipefail

source_if_readable() {
  local file="$1"

  if [[ ! -r "${file}" ]]; then
    return
  fi

  set +e +u
  source "${file}"
  local status=$?
  set -euo pipefail

  if [[ ${status} -ne 0 ]]; then
    echo "Warning: source ${file} exited with ${status}" >&2
  fi
}

if [[ ! -d "${USER_HOME:=$HOME}" ]]; then
  echo "USER_HOME is not set" >&2
  exit 1
fi

if [[ ! -d "${PROJECT_DIR:-}" ]]; then
  echo "PROJECT_DIR is not set or not a directory" >&2
  exit 1
fi

if [[ ! -x "${EXECUTABLE:-}" ]]; then
  echo "Executable not found or not executable: ${EXECUTABLE}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH:-}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

source_if_readable /etc/profile
source_if_readable "${USER_HOME}/.profile"
source_if_readable "${USER_HOME}/.bashrc"

cd "${PROJECT_DIR}"
exec "${EXECUTABLE}" "${CONFIG_PATH}"
