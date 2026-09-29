"""The `opendxp` command.

opendxp export laya|julia|decider|anyjev CHECKPOINT OUT_DIR
opendxp conformance generate PACKAGE --native CHECKPOINT --runtime laya|julia|decider|anyjev
opendxp check PACKAGE [--device cpu|auto|coreml|cuda|openvino|qnn|directml|gpu]
opendxp validate PACKAGE
opendxp run PACKAGE --request request.json
opendxp serve PACKAGE... [--host 127.0.0.1] [--port 8790] [--token TOKEN] [--check]
opendxp mcp PACKAGE...
opendxp bench PACKAGE|URL [--concurrency 1,4,16] [--usd-per-hour PRICE]
opendxp info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from opendxp import __version__


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_export(args: argparse.Namespace) -> int:
    mode = "copy" if args.copy else "link"
    options: dict[str, Any] = {"weights_mode": mode}
    if args.name:
        options["name"] = args.name
    if args.family == "laya":
        from opendxp.export import laya as exporter
    elif args.family == "julia":
        from opendxp.export import julia as exporter

        if args.max_tokens:
            options["max_tokens"] = args.max_tokens
        if args.official_onnx:
            options["official_onnx"] = Path(args.official_onnx)
    elif args.family == "anyjev":
        from opendxp.export import anyjev as exporter

        if not args.gguf:
            raise SystemExit("anyjev: --gguf names the model's GGUF weights")
        options.update(gguf=args.gguf, prior=args.prior, canonical=args.canonical_order)
        if args.base:
            options["base"] = args.base
    else:
        from opendxp.export import decider as exporter

        if args.gguf:
            options["gguf"] = args.gguf
    manifest = exporter.export(Path(args.checkpoint), Path(args.out_dir), **options)
    print(f"wrote {Path(args.out_dir) / 'odxp.json'} ({manifest['profile']})")
    print(
        "next: opendxp conformance generate",
        args.out_dir,
        "--native",
        args.checkpoint,
        "--runtime",
        args.family,
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from opendxp.conformance import DEFAULT_REQUESTS, generate, read_jsonl
    from opendxp.native import load_native

    requests = read_jsonl(Path(args.requests) if args.requests else DEFAULT_REQUESTS)
    if args.limit:
        requests = requests[: args.limit]
    print(f"loading the model's own runtime ({args.runtime}) from {args.native}")
    native = load_native(
        args.runtime, Path(args.native), threads=args.threads, package=Path(args.package)
    )
    try:
        summary = generate(Path(args.package), native, requests)
    finally:
        native.close()
    _print_json(summary)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from opendxp.conformance import check

    options: dict[str, Any] = {}
    if args.optimization:
        options["optimization"] = args.optimization
    if args.batch_rows and args.batch_rows > 1:
        options["batch_rows"] = args.batch_rows
        options["batch_cache"] = args.batch_cache
    report = check(
        Path(args.package),
        device=args.device,
        threads=args.threads,
        hashes=not args.no_hashes,
        **options,
    )
    path = Path(args.report) if args.report else Path(args.package) / f"check-{args.device}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lat = report["latency_ms"]
    agree = report["argmax_agreement"]
    print(f"{report['package']['name']} [{report['package']['profile']}] on {args.device}")
    print(
        f"  cases {report['cases_passed']}/{report['cases']} passed, "
        f"questions {report['questions']}, max |dp| {report['max_abs_dp']:.2e}, "
        f"argmax {agree['agree']}/{agree['total']}, "
        f"errors matched {report['errors']['matched']}/{report['errors']['expected']}"
    )
    print(
        f"  latency: load {lat['load']} ms, request median {lat['request_median']} ms, "
        f"p90 {lat['request_p90']} ms, per question {lat['question_median']} ms"
    )
    for problem in report["package_problems"]:
        print(f"  package: {problem}")
    for failure in report["failures"][:10]:
        print(f"  FAIL {failure['id']}: {failure.get('reason') or failure.get('questions')}")
    verdict = "PASS" if report["passed"] else "FAIL"
    if report["passed"] and not report["compatible"]:
        verdict += " (coverage below the minimum: not badge-eligible)"
    print(f"  {verdict}; report: {path}")
    return 0 if report["passed"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from opendxp.validate import validate_package

    problems = validate_package(Path(args.package), hashes=not args.no_hashes)
    for problem in problems:
        print(problem)
    print("valid" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def cmd_run(args: argparse.Namespace) -> int:
    from opendxp.runtime import load

    text = sys.stdin.read() if args.request == "-" else Path(args.request).read_text()
    request = json.loads(text)
    model = load(Path(args.package), device=args.device, threads=args.threads)
    try:
        _print_json(model.predict(request.get("state", ""), request["questions"]))
    finally:
        model.close()
    return 0


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def cmd_serve(args: argparse.Namespace) -> int:
    from opendxp import spec
    from opendxp.server import serve
    from opendxp.serving import load_models

    token = args.token or os.environ.get("OPENDXP_TOKEN") or None
    models = load_models(
        args.packages, device=args.device, threads=args.threads, check=args.check, log=_stderr
    )
    try:
        server = serve(
            models,
            host=args.host,
            port=args.port,
            token=token,
            cors=args.cors,
            allow_open=args.no_auth,
            quiet=args.quiet,
        )
    except ValueError as exc:
        _stderr(str(exc))
        return 2
    host, port = server.server_address[:2]
    _stderr(
        f"serving {len(models)} model(s) on http://{host}:{port}"
        f" (POST {spec.HTTP_DECIDE_PATH}, GET {spec.HTTP_MODELS_PATH}); Ctrl-C to stop"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for model in models:
            model.close()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from opendxp.mcp import McpServer, protect_stdout, run
    from opendxp.serving import load_models

    protocol = protect_stdout()
    models = load_models(
        args.packages, device=args.device, threads=args.threads, check=args.check, log=_stderr
    )
    _stderr(f"opendxp mcp: {len(models)} model(s) ready on stdio")
    try:
        run(McpServer(models), sys.stdin.buffer, protocol)
    finally:
        for model in models:
            model.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from opendxp.bench import bench_http, bench_package, load_requests, table, token_from_env

    requests = load_requests(Path(args.requests) if args.requests else None, limit=args.limit)
    target = args.target
    if target.startswith(("http://", "https://")):
        levels = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
        extra = {"checkpoint": args.checkpoint} if args.checkpoint else None
        report = bench_http(
            target,
            requests,
            model=args.model,
            extra=extra,
            token=args.token or token_from_env(),
            concurrency=levels,
            duration_s=args.duration,
            warmup=args.warmup,
            usd_per_hour=args.usd_per_hour,
            log=_stderr,
        )
    else:
        report = bench_package(
            Path(target),
            requests,
            device=args.device,
            threads=args.threads,
            rounds=args.rounds,
            warmup=args.warmup,
            usd_per_hour=args.usd_per_hour,
            log=_stderr,
            **(
                {"batch_rows": args.batch_rows, "batch_cache": args.batch_cache}
                if args.batch_rows and args.batch_rows > 1
                else {}
            ),
        )
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(table(report))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from opendxp.providers import describe_machine
    from opendxp.spec import VERSIONS

    _print_json({"opendxp": __version__, "standards": list(VERSIONS), **describe_machine()})
    return 0


def main(argv: list[str] | None = None) -> int:
    from opendxp import spec

    parser = argparse.ArgumentParser(prog="opendxp", description="OpenDXP reference tools")
    parser.add_argument("--version", action="version", version=f"opendxp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="convert a model's checkpoint into an OpenDXP package")
    p.add_argument("family", choices=["laya", "julia", "decider", "anyjev"])
    p.add_argument("checkpoint")
    p.add_argument("out_dir")
    p.add_argument("--name", help="the package name (default: the model's registry name)")
    p.add_argument("--copy", action="store_true", help="copy weights instead of hard-linking")
    p.add_argument("--max-tokens", type=int, help="julia: total token budget (default 8192)")
    p.add_argument(
        "--official-onnx",
        help="julia: map SupersonicLabs/Julia-1-ONNX's model.onnx instead of exporting a graph",
    )
    p.add_argument("--gguf", help="decider, anyjev: the GGUF weights file")
    p.add_argument("--base", help="anyjev: the base model's registry name, e.g. Qwen/Qwen3-1.7B")
    p.add_argument(
        "--prior",
        choices=["none", "content-free"],
        default="none",
        help="anyjev: the label prior divided out of each rotation (default none)",
    )
    p.add_argument(
        "--canonical-order",
        action="store_true",
        help="anyjev: rotate the options sorted by text instead of the request's order",
    )
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("conformance", help="conformance files")
    csub = p.add_subparsers(dest="action", required=True)
    g = csub.add_parser("generate", help="record the model's own answers into conformance.jsonl")
    g.add_argument("package")
    g.add_argument(
        "--native", required=True, help="the checkpoint directory the model's code reads"
    )
    g.add_argument("--runtime", required=True, choices=["laya", "julia", "decider", "anyjev"])
    g.add_argument("--requests", help="a requests JSONL (default: the OpenDXP 0.1 set)")
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--limit", type=int, help="only the first N requests (for a quick try)")
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("check", help="run the conformance file through the reference runtime")
    p.add_argument("package")
    p.add_argument(
        "--device", default="cpu", help="cpu (the reference), auto, coreml, cuda, gpu, ..."
    )
    p.add_argument("--threads", type=int)
    p.add_argument("--report", help="where to write the JSON report")
    p.add_argument("--no-hashes", action="store_true", help="skip the sha256 checks")
    p.add_argument("--optimization", choices=["disable", "basic", "extended", "all"])
    p.add_argument(
        "--batch-rows", type=int, help="causal-letters: decode this many rows together (serving)"
    )
    p.add_argument("--batch-cache", choices=["shared", "per-row"], default="shared")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("validate", help="check a package's files against the schemas and hashes")
    p.add_argument("package")
    p.add_argument("--no-hashes", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="answer one request with a package")
    p.add_argument("package")
    p.add_argument("--request", required=True, help="a request JSON file, or - for stdin")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="serve packages over the HTTP binding (SPEC.md 11)")
    p.add_argument("packages", nargs="+", metavar="PACKAGE")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=spec.DEFAULT_PORT)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int)
    p.add_argument("--token", help="require this bearer token (or set OPENDXP_TOKEN)")
    p.add_argument(
        "--no-auth",
        action="store_true",
        help="listen beyond this machine without a token (only behind an authenticating proxy)",
    )
    p.add_argument("--cors", action="append", help="an origin browsers may call from (or *)")
    p.add_argument(
        "--check", action="store_true", help="run each conformance file first and report it"
    )
    p.add_argument("--quiet", action="store_true", help="no access log")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="serve packages as MCP tools on stdio (SPEC.md 12)")
    p.add_argument("packages", nargs="+", metavar="PACKAGE")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int)
    p.add_argument(
        "--check", action="store_true", help="run each conformance file first and report it"
    )
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("bench", help="how fast a package or an OpenDXP server answers")
    p.add_argument("target", help="a package directory, or a server's URL (http://host:port)")
    p.add_argument("--requests", help="a requests JSONL (default: the OpenDXP 0.1 set)")
    p.add_argument("--limit", type=int, help="only the first N requests")
    p.add_argument("--model", help="the model to ask, for a server that holds several")
    p.add_argument("--checkpoint", help="the checkpoint to ask, for an engine that takes one")
    p.add_argument("--token", help="a bearer token for the server (or set OPENDXP_TOKEN)")
    p.add_argument("--concurrency", default="1,4,16", help="clients at once, per level (servers)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds per level (servers)")
    p.add_argument("--rounds", type=int, default=1, help="passes over the requests (packages)")
    p.add_argument("--warmup", type=int, default=3, help="requests sent before timing starts")
    p.add_argument("--device", default="auto", help="the device for a package")
    p.add_argument("--threads", type=int)
    p.add_argument(
        "--batch-rows", type=int, help="causal-letters: decode this many rows together (packages)"
    )
    p.add_argument("--batch-cache", choices=["shared", "per-row"], default="shared")
    p.add_argument("--usd-per-hour", type=float, help="the machine's price, for the cost per 1,000")
    p.add_argument("--report", help="where to write the JSON report")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("info", help="this machine's backends and devices")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
