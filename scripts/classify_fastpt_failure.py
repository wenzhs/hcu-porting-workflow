#!/usr/bin/env python3
"""Route-aware diagnosis of the first error, without cleanup or activation.

The legacy classify(text) -> (classification, first_error, action) API and CLI
text keys are retained. A classification is advice, not a build success receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptation_common import AdaptationError, ROUTES, read_text

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR_LINE = re.compile(r"(?:fatal\s+)?error:|CMake Error|undefined reference|undefined symbol|cannot find -l|No such file|ImportError:|ModuleNotFoundError:|PermissionError:|Permission denied|TimeoutError:|timed out", re.I)


def first_error_block(text: str) -> tuple[str, str]:
    lines = ANSI.sub("", text).splitlines()
    for index, line in enumerate(lines):
        if ERROR_LINE.search(line):
            block = [line.strip()]
            # Forward context only. Banners and unrelated earlier paths cannot win.
            for following in lines[index + 1:index + 13]:
                if not following.strip():
                    if len(block) > 1:
                        break
                    continue
                if ERROR_LINE.search(following):
                    break
                block.append(following.strip())
            return line.strip(), "\n".join(block)
    return "", ""


def classify(text: str, route: str | None = None, phase: str | None = None) -> tuple[str, str, str]:
    if route == "cutlass-hytlass":
        route = "transcode"
    if route is not None and route not in ROUTES:
        raise AdaptationError(f"Invalid route: {route}")
    first, block = first_error_block(text)
    if not first:
        return "unknown", "(未找到明确 error 行)", "保留进程真实退出码和完整日志；没有error行不代表执行成功。"
    lower = block.lower()
    if re.search(r"torch version.*(?:does not match|required|not supported)|fastpt.*version.*mismatch", block, re.I):
        advice = "核对已安装版本及目标框架分派；不要移除版本检查、自动降级或激活另一条路线。"
        if route == "native-hip":
            advice = "native出现FastPT版本错误，应核查实际Extension/HIPIFY是否意外委托FastPT；不自动修复或激活环境。"
        return "fastpt-version", first, advice
    if re.search(r"PermissionError:|Permission denied", block, re.I):
        return "permission", first, "记录所需路径和权限，由用户处理；不提权、不改权限、不删除其他任务文件。"
    if re.search(r"TimeoutError:|timed out", block, re.I):
        return "timeout", first, "区分无进展与长编译，保存超时和进程状态；不把超时记为成功或无限重试。"
    if route == "non-transcode" and re.search(r"cannot find\s+-lcudart_static|(?:error:|undefined reference).*cudart_static", block, re.I):
        return "static-cudart", first, "核对本版本GPUFusion静态链接限制和用户交付约束；仅在允许动态链接时调整配置，不静默改变要求。"
    if re.search(r"fatal error:.*(?:file not found|No such file)|cannot open include file|No such file or directory", block, re.I):
        return "missing-header", first, "核对当前路线的include路径、声明依赖和真实缺失文件；不要按.hip后缀清理源码或自动加载FastPT。"
    if re.search(r"undefined reference|undefined symbol|cannot find\s+-l", block, re.I):
        return "missing-symbol", first, "核对本路线实际链接/加载库、ABI和产物身份；保留失败符号，不自动换路或安装库。"
    if re.search(r"ModuleNotFoundError:|ImportError:", block):
        return "python-extension", first, "检查当前解释器、构建隔离和已声明依赖；确认加载的是本次产物，不自动卸载或覆盖框架。"
    if "cmake error" in lower or re.search(r"No CMAKE_(?:CUDA|HIP|CXX)_COMPILER|Unknown CMake command", block, re.I):
        return "cmake-target", first, "检查当前target的原构建入口、编译器、架构和package作用域；不全量改写CMake或引入另一条路线。"
    if re.search(r"redefinition.*\bdim3\b|\bdim3\b.*redefinition", block, re.I):
        if route == "non-transcode" and "hip_runtime" in lower and re.search(r"cuda(?:_runtime|/|\\)", lower):
            return "transcode-artifact-contamination", first, "疑似同一target混入两种头文件；检查真实include链和产物来源，不根据文件名删除源码。"
        return "cuda-hip-type-conflict", first, "核对该target的声明来源、include顺序和所选backend；重定义本身不证明所有HIP文件都是污染。"
    if re.search(r"(?:cannot initialize|no viable conversion|no matching function).*\b(?:hip|cuda)|hipDataType|hipblasDataType", block, re.I):
        return "cuda-hip-type-conflict", first, "按实际库版本核对参数类型和转换结果；不靠强制cast或混入其他路线头文件掩盖差异。"
    if re.search(r"__shfl\w*_sync|__ballot_sync|__any_sync|__all_sync|__syncwarp", block):
        return "warp-assumption", first, "按目标编译器核对intrinsic、mask、逻辑分组和同步语义；不统一删除_sync或将32改64。"
    if re.search(r"(?:unsupported|not supported|static assertion|not declared|undefined).*(?:cutlass|wgmma|tcgen05|hytlass)|(?:cutlass|hytlass).*static assertion", block, re.I):
        return "cutlass-unsupported", first, "保留原始架构/模板风险；按目标功能和固定库版本判断可替代实现，重大重写或裁剪需确认。"
    return "unknown", first, "按首个错误保留最小复现、路线与阶段；现有证据不足以自动归因，不执行破坏性修复。"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Build/runtime log path")
    parser.add_argument("--route", choices=(*ROUTES, "cutlass-hytlass"))
    parser.add_argument("--phase", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        name, first, action = classify(read_text(args.log), args.route, args.phase)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    payload = {"classification": name, "first_error": first, "action": action}
    if args.json:
        print(json.dumps({**payload, "route": args.route, "phase": args.phase, "executed_fix": False}, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
