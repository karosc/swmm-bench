from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil  # pyright: ignore[reportMissingModuleSource]
from swmm.pandas import Report  # pyright: ignore[reportMissingImports]

from swmm_bench.models import EngineResult


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _display_name(inp_path: Path) -> str:
    try:
        return str(inp_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return inp_path.name


def _analysis_duration_seconds(rpt_path: Path) -> float | None:
    try:
        return Report(str(rpt_path)).analysis_duration.total_seconds()
    except Exception:
        # A run can still produce a useful partial report without timing metadata.
        return None


def _copy_model_tree(
    inp_path: Path,
    destination_root: Path,
    excluded_roots: tuple[Path, ...] = (),
) -> Path:
    source_root = inp_path.parent.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    copied_inp = destination_root / inp_path.name
    excluded_paths = {path.expanduser().resolve() for path in excluded_roots}
    excluded_paths.add(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    def is_excluded(path: Path) -> bool:
        absolute_path = path.absolute()
        return any(
            absolute_path == excluded_path or excluded_path in absolute_path.parents
            for excluded_path in excluded_paths
        )

    for directory, directory_names, file_names in os.walk(
        source_root,
        topdown=True,
        followlinks=False,
    ):
        source_directory = Path(directory)
        target_directory = destination_root / source_directory.relative_to(source_root)
        target_directory.mkdir(parents=True, exist_ok=True)

        directory_names[:] = [
            name for name in directory_names if not is_excluded(source_directory / name)
        ]
        for name in directory_names:
            (target_directory / name).mkdir(parents=True, exist_ok=True)

        for name in file_names:
            source = source_directory / name
            if not is_excluded(source):
                shutil.copy2(source, target_directory / name)

    if not copied_inp.exists():
        shutil.copy2(inp_path, copied_inp)

    return copied_inp


def _set_option(inp_path: Path, option_name: str, value: int | float) -> None:
    with inp_path.open("r", encoding="utf-8", newline="") as inp_file:
        text = inp_file.read()
    options = re.search(
        r"(?ims)^\[OPTIONS\][ \t]*(?:;[^\r\n]*)?(?=\r?$).*?(?=^\[|\Z)",
        text,
    )
    if options is None:
        return
    section, replacements = re.subn(
        rf"(?im)^([ \t]*{re.escape(option_name)}[ \t]+)\S+"
        r"([ \t]*(?:;[^\r\n]*)?)(?=\r?$)",
        rf"\g<1>{value}\g<2>",
        options.group(),
    )
    if replacements == 0:
        header = re.search(
            r"(?im)^\[OPTIONS\][ \t]*(?:;[^\r\n]*)?(?=\r?$)",
            section,
        )
        if header is None:
            raise ValueError("Matched [OPTIONS] section has no valid section header")
        newline = "\r\n" if "\r\n" in text else "\n"
        section = (
            section[: header.end()]
            + f"{newline}{option_name:<20} {value}"
            + section[header.end() :]
        )
    with inp_path.open("w", encoding="utf-8", newline="") as inp_file:
        inp_file.write(text[: options.start()] + section + text[options.end() :])


def _set_threads(inp_path: Path, threads: int) -> None:
    _set_option(inp_path, "THREADS", threads)


def _set_variable_step(inp_path: Path, variable_step: float) -> None:
    _set_option(inp_path, "VARIABLE_STEP", variable_step)


def run_engine(
    engine_path: str,
    inp_path: Path,
    work_dir: Path,
    timeout: float | None,
    inp_name: str | None = None,
    inp_identity: str | None = None,
    engine_name: str | None = None,
    threads: int = 1,
    variable_step: float | None = None,
    excluded_roots: tuple[Path, ...] = (),
) -> EngineResult:
    engine = Path(engine_path).expanduser().resolve()
    resolved_inp = inp_path.expanduser().resolve()
    display_name = inp_name or _display_name(resolved_inp)
    engine_name = engine_name or engine.name
    case_dir = work_dir / _safe_name(engine_name) / _safe_name(display_name)
    case_dir.mkdir(parents=True, exist_ok=True)

    copied_inp = _copy_model_tree(
        resolved_inp,
        case_dir / "model",
        excluded_roots=excluded_roots,
    )
    _set_threads(copied_inp, threads)
    if variable_step is not None:
        _set_variable_step(copied_inp, variable_step)
    execution_cwd = case_dir / "model"

    rpt_candidate = case_dir / "raw.rpt"
    out_candidate = case_dir / "raw.out"
    final_rpt = case_dir / "result.rpt"
    final_out = case_dir / "result.out"

    peak_rss_bytes = 0
    process_done = threading.Event()

    def poll_memory(process: psutil.Process) -> None:
        nonlocal peak_rss_bytes
        while not process_done.is_set():
            try:
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    rss += child.memory_info().rss
                peak_rss_bytes = max(peak_rss_bytes, rss)
            except (psutil.Error, ProcessLookupError):
                if process_done.is_set():
                    break
            time.sleep(0.1)

    stdout = ""
    stderr = ""
    exit_code: int | None = None
    error: str | None = None
    memory_thread: threading.Thread | None = None

    try:
        for artifact in (rpt_candidate, out_candidate, final_rpt, final_out):
            if artifact.exists() or artifact.is_symlink():
                if artifact.is_symlink() or not artifact.is_file():
                    raise RuntimeError(f"Expected output artifact file at {artifact}")
                artifact.unlink()
        process = subprocess.Popen(
            [str(engine), str(copied_inp), str(rpt_candidate), str(out_candidate)],
            cwd=str(execution_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ps_process = psutil.Process(process.pid)
        memory_thread = threading.Thread(
            target=poll_memory, args=(ps_process,), daemon=True
        )
        memory_thread.start()

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        # pi-lens-ignore: ast-grep:no-bare-except
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            exit_code = process.returncode
            error = f"Timed out after {timeout} seconds"
    except OSError as exc:
        error = str(exc)
    finally:
        process_done.set()
        if memory_thread is not None:
            memory_thread.join(timeout=1.0)

    duration_s: float | None = None
    peak_memory_mb = peak_rss_bytes / (1024 * 1024) if peak_rss_bytes else None
    rpt_path: str | None = None
    out_path: str | None = None

    if rpt_candidate.exists() and rpt_candidate.stat().st_size > 0:
        shutil.copy2(rpt_candidate, final_rpt)
        rpt_path = str(final_rpt.resolve())
        duration_s = _analysis_duration_seconds(final_rpt)
    elif error is None:
        error = "Engine did not produce a non-empty report file"

    if out_candidate.exists() and out_candidate.stat().st_size > 0:
        shutil.copy2(out_candidate, final_out)
        out_path = str(final_out.resolve())

    return EngineResult(
        engine_path=str(engine),
        engine_name=engine_name,
        inp_path=inp_identity or str(resolved_inp),
        inp_name=display_name,
        duration_s=duration_s,
        peak_memory_mb=peak_memory_mb,
        exit_code=exit_code,
        rpt_path=rpt_path,
        stdout=stdout,
        stderr=stderr,
        error=error,
        out_path=out_path,
    )


def run_benchmark(
    engines: list[str],
    inp_files: list[Path],
    work_dir: Path,
    benchmark_name: str,
    timeout: float | None,
    threads: int = 1,
    variable_step: float | None = None,
    progress_callback: Callable[[EngineResult], None] | None = None,
    inp_names: dict[Path, str] | None = None,
    inp_identities: dict[Path, str] | None = None,
) -> list[EngineResult]:
    results: list[EngineResult] = []
    resolved_names = {
        path.expanduser().resolve(): name for path, name in (inp_names or {}).items()
    }
    resolved_identities = {
        path.expanduser().resolve(): identity
        for path, identity in (inp_identities or {}).items()
    }
    labelled_inputs = [
        (
            resolved_names.get(path.expanduser().resolve(), _display_name(path)),
            path,
            resolved_identities.get(path.expanduser().resolve()),
        )
        for path in inp_files
    ]

    benchmark_dir = work_dir / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    for engine_path in engines:
        for display_name, inp_path, inp_identity in labelled_inputs:
            result = run_engine(
                engine_path=engine_path,
                inp_path=inp_path,
                work_dir=benchmark_dir,
                timeout=timeout,
                inp_name=display_name,
                inp_identity=inp_identity,
                threads=threads,
                variable_step=variable_step,
                excluded_roots=(work_dir,),
            )
            results.append(result)
            if progress_callback is not None:
                progress_callback(result)

    return results
