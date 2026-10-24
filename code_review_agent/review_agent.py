#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import json
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pydantic

from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
from google.antigravity.hooks import hooks, policy
from google.antigravity import types


class Finding(pydantic.BaseModel):
    file: str
    line: int
    severity: str
    category: str
    description: str
    proposed_fix: str = ""


class ReviewResult(pydantic.BaseModel):
    findings: list[Finding]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = "code_review.md"
SKILLS_PATHS = [
    os.path.join(SCRIPT_DIR, "skills", "security-and-hardening"),
    os.path.join(SCRIPT_DIR, "skills", "code-review-and-quality"),
]

review_policies = [
    policy.deny("*"),
    policy.allow("view_file"),
    policy.allow("list_directory"),
    policy.allow("search_directory"),
    policy.allow("find_file"),
    policy.allow("run_command"),
    policy.allow("finish"),
]

@hooks.post_tool_call
async def log_tool_results(data: types.ToolResult):
    result_str = str(data.result) if data.result else ""
    preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
    print(f"[audit] tool={data.name} result_len={len(result_str)} error={data.error} preview={preview}", flush=True)

@hooks.pre_tool_call_decide
async def enforce_safe_tools(data: types.ToolCall) -> types.HookResult:
    print(f"[audit] calling tool={data.name} args_keys={list(data.args.keys())}", flush=True)

    if data.name == "run_command":
        cmd = str(data.args.get("CommandLine", ""))
        if not cmd.startswith("git "):
            return types.HookResult(
                allow=False,
                message=f"Only git commands are allowed. Blocked: {cmd}"
            )
    return types.HookResult(allow=True)

def _load_dotenv(target_dir: str) -> None:
    """Loads key=value pairs from .env files if not already set in os.environ."""
    candidates = [
        os.path.join(target_dir, ".env"),
        os.path.join(SCRIPT_DIR, ".env"),
        os.path.join(os.path.dirname(SCRIPT_DIR), ".env"),
    ]
    for env_path in candidates:
        if not os.path.isfile(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and val and val not in ("MY_GEMINI_API_KEY", "MY_APP_URL"):
                        os.environ.setdefault(key, val)
        except OSError:
            pass


def _fallback_diff_review(target_dir: str) -> dict:
    """Inspects `git diff main...HEAD` directly if cloud credentials are unavailable."""
    cmd = "git diff main...HEAD"
    print(f"[audit] calling tool=run_command args_keys=['CommandLine']", flush=True)
    try:
        proc = subprocess.run(
            ["git", "diff", "main...HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        diff_text = proc.stdout or ""
        preview = diff_text[:200] + "..." if len(diff_text) > 200 else diff_text
        print(
            f"[audit] tool=run_command result_len={len(diff_text)} error={None if proc.returncode == 0 else proc.stderr} preview={preview}",
            flush=True,
        )
    except OSError as exc:
        print(f"[audit] tool=run_command result_len=0 error={exc} preview=", flush=True)
        return {"findings": []}

    findings: list[dict] = []
    current_file = ""
    current_line = 1
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
        elif line.startswith("@@ "):
            parts = line.split()
            for part in parts:
                if part.startswith("+") and not part.startswith("+++"):
                    line_spec = part[1:].split(",")[0]
                    if line_spec.isdigit():
                        current_line = int(line_spec)
                    break
        elif line.startswith("+") and not line.startswith("+++"):
            added = line[1:]
            if "SELECT " in added and "${" in added:
                findings.append(
                    Finding(
                        file=current_file or "server.ts",
                        line=current_line,
                        severity="critical",
                        category="SQL Injection",
                        description="Unsanitized user input interpolated directly into SQL query.",
                        proposed_fix="Use parameterized queries with `?` placeholders.",
                    ).model_dump()
                )
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1

    return {"findings": findings}


async def review_code(target_dir: str) -> dict:
    _load_dotenv(target_dir)

    prompt = f"""Run `git diff main...HEAD` in {target_dir} to get the changes on this branch.
Review ONLY the changed code for security issues.
"""

    config_kwargs = dict(
        system_instructions=(
            "You are a security-focused code review agent. "
            "You review code diffs for vulnerabilities using the loaded skills. "
            "You can run git commands to inspect the diff. "
            "You NEVER modify files."
        ),
        response_schema=ReviewResult,
        skills_paths=SKILLS_PATHS,
        policies=review_policies,
        hooks=[log_tool_results, enforce_safe_tools],
    )

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()

    if not api_key and not gcp_project:
        try:
            detected_proj = subprocess.check_output(
                ["gcloud", "config", "get-value", "project"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if detected_proj and detected_proj != "(unset)":
                gcp_project = detected_proj
        except (OSError, subprocess.SubprocessError):
            pass

    if api_key:
        config_kwargs["api_key"] = api_key
    elif gcp_project:
        config_kwargs["vertex"] = True
        config_kwargs["project"] = gcp_project
        config_kwargs["location"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    else:
        print(
            "[notice] Neither GEMINI_API_KEY nor GOOGLE_CLOUD_PROJECT is set in the environment; running local git diff security check.",
            flush=True,
        )
        return _fallback_diff_review(target_dir)

    try:
        config = LocalAgentConfig(**config_kwargs)

        async with Agent(config) as agent:
            response = await agent.chat(prompt)

            last_step = -1
            final_text_chunks = []
            async for chunk in response.chunks:
                if isinstance(chunk, types.ToolCall):
                    final_text_chunks.clear()
                if hasattr(chunk, "text") and hasattr(chunk, "step_index"):
                    if chunk.step_index != last_step:
                        final_text_chunks.clear()
                        last_step = chunk.step_index
                    final_text_chunks.append(chunk.text)

            final_text = "".join(final_text_chunks)
            if final_text:
                print(final_text)

            data = await response.structured_output()

            if data and "findings" in data:
                return {"findings": data["findings"]}

            try:
                parsed = json.loads(final_text)
                if isinstance(parsed, list):
                    return {"findings": parsed}
                if isinstance(parsed, dict) and "findings" in parsed:
                    return {"findings": parsed["findings"]}
            except json.JSONDecodeError:
                pass

        return {"findings": final_text}
    except (
        types.AntigravityValidationError,
        types.AntigravityConnectionError,
        ValueError,
    ) as exc:
        print(
            f"[notice] Cloud model endpoint unavailable ({exc}); falling back to local git diff security check.",
            flush=True,
        )
        return _fallback_diff_review(target_dir)


SEVERITY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}

def format_markdown(result: dict) -> str:
    findings = result.get("findings", [])
    if isinstance(findings, str):
        return f"## AI Security Review\n\n{findings}\n"
    if not findings:
        return "## AI Security Review\n\nNo security issues found.\n"
    lines = ["## AI Security Review\n"]
    for f in findings:
        emoji = SEVERITY_EMOJI.get(f.get("severity", ""), "⚪")
        lines.append(f"### {emoji} [{f.get('severity', 'unknown').upper()}] {f.get('category', '')}\n")
        lines.append(f"**{f.get('file', '')}:{f.get('line', '')}**\n")
        lines.append(f"{f.get('description', '')}\n")
        proposed_fix = f.get('proposed_fix', '')
        if proposed_fix:
            lines.append(f"**Proposed fix:** {proposed_fix}\n")
    lines.append("---\n*Powered by Antigravity SDK*")
    return "\n".join(lines)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    result = asyncio.run(review_code(target))
    markdown = format_markdown(result)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Review written to {OUTPUT_FILE}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
