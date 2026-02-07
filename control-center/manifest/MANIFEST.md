# Service Manifest

The control center reads a JSON manifest that defines service launch commands, health checks, and log file targets.

## Top-Level Fields

- `schema_version` (`Int`): manifest schema version.
- `app_name` (`String`): display name for the control center.
- `variables` (`Object<String,String>`): custom variables for token expansion.
- `default_log_directory` (`String`): default folder for relative `log_file` values.
- `services` (`Array<Service>`): service definitions.

## Service Fields

- `id` (`String`): stable unique id.
- `name` (`String`): UI label.
- `working_directory` (`String`): process working dir.
- `executable` (`String`): executable path.
- `arguments` (`Array<String>`): process arguments.
- `environment` (`Object<String,String>`): extra env vars.
- `log_file` (`String`): absolute path or file name relative to `default_log_directory`.
- `health_check` (`Object`): health probe config.
- `bootstrap` (`Object`, optional): environment bootstrap instructions for in-app venv rebuild.
- `auto_start` (`Bool`): start automatically at app launch.
- `restart_on_crash` (`Bool`): restart if unexpected exit.
- `graceful_shutdown_seconds` (`Int`): wait time before force-kill.

## `health_check` Fields

- `url` (`String`): endpoint URL.
- `expected_status` (`Int`): expected HTTP status.
- `interval_seconds` (`Int`): poll interval.
- `timeout_seconds` (`Int`): per-request timeout.

## `bootstrap` Fields

- `python_executable` (`String`, default `python3`): interpreter used to create venv (`python -m venv`). For this project use `python3.11`.
- `venv_directory` (`String`): target `.venv` directory path.
- `requirements_file` (`String`): requirements file passed to `pip install -r`.
- `upgrade_build_tools` (`Bool`, default `true`): run `pip install --upgrade pip setuptools wheel`.
- `pip_arguments` (`Array<String>`, default `[]`): extra arguments appended to `pip install`.

## Variable Expansion

The loader supports `${TOKEN}` replacement in path, arg, and env fields.

Built-in tokens:
- `${HOME}`
- `${TMPDIR}`
- `${MANIFEST_DIR}`
- `${WORKSPACE_ROOT}`
- `${APP_SUPPORT_DIR}`

You can define additional tokens in `variables`.

## Recommended Production Pattern

For installer builds, copy a generated manifest to:

- `~/Library/Application Support/GaryLocalhost/manifest/services.json`

Then launch the app with:

- `GARY_SERVICE_MANIFEST=/path/to/services.json`

so runtime paths are fully explicit and independent of source checkout locations.
