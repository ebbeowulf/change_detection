# Need to define CHANGE_HOME environment variable before running this script
#   CHANGE_HOME should point to the root directory of the change_detection repository
is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "${0}" ]]
}
if [ -z "${CHANGE_HOME-}" ]; then
  echo "ERROR: required environment variable CHANGE_HOME is not set. Aborting." >&2
  if is_sourced; then
    return 1
  else
    exit 1
  fi
fi