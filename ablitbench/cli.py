from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, get_args

import click
import yaml
from rich.console import Console

from ablitbench.core import REGISTRY, Scanner
from ablitbench.formats.json import loads as json_loads
from ablitbench.formats.yaml import load as yaml_load
from ablitbench.models import Event, Protocol, ScanConfig
from ablitbench.report import BANNER, banner, event_terminal, terminal


class BannerCommand(click.Command):
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write(BANNER + "\n\n")
        super().format_help(ctx, formatter)


class BannerGroup(click.Group):
    command_class = BannerCommand

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write(BANNER + "\n\n")
        super().format_help(ctx, formatter)


@click.group(cls=BannerGroup)
@click.version_option(package_name="ablitbench")
def main() -> None:
    pass


def configuration_schema(ctx: click.Context, parameter: click.Parameter, value: bool) -> None:
    if value and not ctx.resilient_parsing:
        click.echo(json.dumps(ScanConfig.model_json_schema(), indent=2))
        ctx.exit()


def research_operations(ctx: click.Context, parameter: click.Parameter, value: bool) -> None:
    if value and not ctx.resilient_parsing:
        from ablitbench.parity.catalog import operations
        from ablitbench.parity.sources import catalog, source_identity

        click.echo(
            json.dumps(
                {
                    "projects": {
                        project: {
                            "source_identity": source_identity(project),
                            "operations": operations(project),
                        }
                        for project in catalog()["repositories"]
                    },
                    "verification": "Feature verification is separate from operation availability",
                },
                indent=2,
            )
        )
        ctx.exit()


def common(function: Any) -> Any:
    options = [
        click.argument("target", required=False),
        click.option(
            "--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
        ),
        click.option(
            "--schema",
            is_flag=True,
            is_eager=True,
            expose_value=False,
            callback=configuration_schema,
            help="Print the configuration JSON Schema without making requests.",
        ),
        click.option(
            "--operations",
            is_flag=True,
            is_eager=True,
            expose_value=False,
            callback=research_operations,
            help="Print upstream operation contracts without execution.",
        ),
        click.option("--checks", help="Comma-separated IDs: " + ", ".join(s.id for s in REGISTRY)),
        click.option("--output", type=click.Path(file_okay=False, path_type=Path)),
        click.option(
            "--format",
            "output_format",
            type=click.Choice(["terminal", "json", "jsonl"]),
            default="terminal",
        ),
        click.option("--plan", "plan_only", is_flag=True),
        click.option("--resume", type=click.Path(exists=True, file_okay=False, path_type=Path)),
        click.option("--replay", type=click.Path(exists=True, file_okay=False, path_type=Path)),
        click.option("--seed", type=int),
        click.option("--execute-tools", is_flag=True, default=None),
    ]
    for option in options:
        function = option(function)
    return function


@main.command(
    help=(
        "Assess an API endpoint. Default checks send inference requests and "
        "may incur provider costs."
    )
)
@common
@click.option("--model")
@click.option("--key", help="API key for this invocation.")
@click.option(
    "--protocol",
    type=click.Choice(get_args(Protocol)),
)
def api(**arguments: Any) -> None:
    invoke("api", arguments)


@main.command(
    help=(
        "Assess source without importing model code. Inference and real "
        "tools require configured sandbox images."
    )
)
@common
@click.option("--revision")
@click.option("--reference")
@click.option("--offline", is_flag=True, default=None)
@click.option("--inference", is_flag=True, default=None)
def source(**arguments: Any) -> None:
    invoke("source", arguments)


def configuration(mode: str, arguments: dict[str, Any]) -> ScanConfig:
    data: dict[str, Any] = {}
    path = arguments.get("config_path")
    if path:
        text = path.read_text(encoding="utf-8")
        loaded = json_loads(text) if path.suffix.lower() == ".json" else yaml_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("Config must be an object")
        data = loaded
    if data.get("mode", mode) != mode:
        raise ValueError("Config mode conflicts with command")
    data["mode"] = mode
    target = arguments.get("target")
    if target and "targets" in data:
        raise ValueError("Positional target conflicts with config targets")
    if target:
        data["targets"] = [{"location": target}]
    if not data.get("targets"):
        raise ValueError("Supply a target or --config with targets")
    target_fields = {"model", "protocol", "revision", "reference"}
    for key in target_fields:
        if arguments.get(key) is not None:
            for entry in data["targets"]:
                entry[key] = arguments[key]
    for key in ("output", "resume", "replay", "seed", "offline", "inference", "execute_tools"):
        if arguments.get(key) is not None:
            data[key] = arguments[key]
    if arguments.get("checks") is not None:
        data["checks"] = [
            value.strip() for value in arguments["checks"].split(",") if value.strip()
        ]
    config = ScanConfig.model_validate(data)
    if arguments.get("key") is not None:
        from ablitbench.parity.native_provider import origin

        if len({origin(entry.location) for entry in config.targets}) != 1:
            raise ValueError("--key requires targets on one API origin")
        for entry in config.targets:
            entry.bind_key(arguments["key"])
    return config


def invoke(mode: str, arguments: dict[str, Any]) -> None:
    output_format = arguments["output_format"]
    console = Console(stderr=True)
    try:
        config = configuration(mode, arguments)
        if output_format == "terminal" and not arguments["plan_only"]:
            banner(console)

        def on_event(event: Event) -> None:
            if output_format == "jsonl":
                click.echo(event.model_dump_json())
            elif output_format == "terminal":
                event_terminal(console, event)

        async def operation() -> Any:
            scanner = Scanner()
            if arguments["plan_only"]:
                return await scanner.plan(config)
            return await scanner.run(config, on_event=on_event)

        result = asyncio.run(operation())
        if arguments["plan_only"]:
            click.echo(result.model_dump_json(indent=2))
            import shutil

            shutil.rmtree(result.inventory_dir)
            return
        if output_format == "json":
            with (result.directory / "report.json").open(encoding="utf-8") as stream:
                import shutil

                shutil.copyfileobj(stream, sys.stdout, length=1024 * 1024)
                click.echo()
        elif output_format == "terminal":
            terminal(console, result)
        code = (
            130
            if result.status == "cancelled"
            else 3
            if result.status == "partial"
            else 1
            if any(f.severity in {"medium", "high", "critical"} for f in result.findings())
            else 0
        )
        raise click.exceptions.Exit(code)
    except (ValueError, OSError, sqlite3.Error, yaml.YAMLError) as error:
        raise click.UsageError(str(error)) from error
    except KeyboardInterrupt:
        raise click.exceptions.Exit(130) from None


if __name__ == "__main__":
    main()
