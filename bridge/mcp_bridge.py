"""
CE-MCP-Plugin 桥接服务器（Streamable HTTP MCP Server）
========================================================

本模块实现一个 MCP（Model Context Protocol）桥接服务器，用于把 Cheat Engine
插件 CE-MCP-Plugin 暴露的 75 个（配置实际为 77 个）命令封装成 MCP tools，
供支持 MCP 的 AI 客户端调用。

协议概述
--------
- CE-MCP-Plugin 插件作为 TCP **客户端**，启动后自动连接本机 127.0.0.1:8888
  （见 /workspace/CE-MCP-Plugin/mcp.json 中的 mcp.server 字段）。
- 命令通过 TCP 以文本行方式交互：
  * 发送：一行文本命令，以 ``\\n`` 结尾，形如 ``COMMAND:param1,param2`` 或
    无参数命令 ``COMMAND``。
  * 接收：插件执行后回传一行结果文本，以 ``\\n`` 结尾。
- 本桥是一个 Streamable HTTP MCP 服务器，监听 127.0.0.1:8080，客户端连接
  URL 为 http://127.0.0.1:8080/mcp。

职责
----
1. 读取 mcp.json 配置，收集全部命令（名称、中文描述、语法）。
2. 从命令语法（syntax）解析参数名。
3. 为每个命令动态注册一个 MCP tool。
4. 通过 TCP 客户端把 MCP 调用转发给 CE 插件，并返回插件的文本响应。

依赖
----
- Python >= 3.10
- 标准库 + 官方 ``mcp`` 包（``from mcp.server import MCPServer``）。
"""

import json
import socket
import threading
from typing import Annotated, Optional

# pydantic 的 Field，用于给参数补充简短中文描述（mcp 包依赖 pydantic）
from pydantic import Field

from mcp.server import MCPServer

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

# CE-MCP-Plugin 配置文件的绝对路径
CONFIG_PATH = "/workspace/CE-MCP-Plugin/mcp.json"


def _load_config(path: str = CONFIG_PATH) -> dict:
    """读取 CE-MCP-Plugin 的 mcp.json 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_CONFIG = _load_config()

# 插件 TCP 服务器的地址与端口
HOST = _CONFIG.get("mcp", {}).get("server", {}).get("host", "127.0.0.1")
PORT = _CONFIG.get("mcp", {}).get("server", {}).get("port", 8888)

# 连接与接收超时（秒）
SOCKET_TIMEOUT = 10.0

# 桥服务器的监听地址与端口
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8080


# ---------------------------------------------------------------------------
# 参数名解析（从 syntax 提取）
# ---------------------------------------------------------------------------

def _clean_param_segment(seg: str) -> str:
    """清洗单个参数段，返回清洗后的参数名（可能为空字符串）。

    规则：
    - 去除 ``[`` ``]`` ``<`` ``>`` ``(`` ``)`` 及两端空白。
    - 去除结尾的 ``...``。
    - 去除 ``|`` 及之后的内容（形如 ``a|b`` 取 ``a``）。
    """
    seg = seg.strip()
    for ch in "[]<>()":
        seg = seg.replace(ch, "")
    seg = seg.strip()
    if seg.endswith("..."):
        seg = seg[:-3].rstrip()
    if "|" in seg:
        seg = seg.split("|", 1)[0].rstrip()
    return seg


def _parse_params(syntax: str) -> list:
    """从命令语法解析参数名列表。

    - syntax 形如 ``COMMAND:p1,p2,p3`` 或 ``COMMAND``（无参数）。
    - 无冒号 -> 返回空列表。
    - 冒号后按逗号切分，逐段清洗；清洗后为空的段跳过。
    """
    if ":" not in syntax:
        return []
    rest = syntax.split(":", 1)[1]
    params = []
    for seg in rest.split(","):
        cleaned = _clean_param_segment(seg)
        if cleaned:
            params.append(cleaned)
    return params


# ---------------------------------------------------------------------------
# 命令收集
# ---------------------------------------------------------------------------

def _collect_commands(config: dict) -> list:
    """把 mcp.json 中 commands.categories 下的命令全部收拢为扁平列表。

    每条命令对象含字段：name、description、syntax、example。
    """
    flat = []
    categories = config.get("commands", {}).get("categories", [])
    for cat in categories:
        flat.extend(cat.get("commands", []))
    return flat


_COMMANDS = _collect_commands(_CONFIG)


# ---------------------------------------------------------------------------
# TCP 客户端（连接 CE 插件）
# ---------------------------------------------------------------------------

# 全局 socket，模块级维护，由 _lock 串行化所有请求
_sock: Optional[socket.socket] = None
_connected: bool = False
# 串行化所有请求：插件同时只接受一个连接、一次处理一个命令
_lock = threading.Lock()


def _close_socket() -> None:
    """关闭并清空全局 socket。"""
    global _sock, _connected
    if _sock is not None:
        try:
            _sock.close()
        except Exception:
            pass
    _sock = None
    _connected = False


def _ensure_connected() -> bool:
    """确保已建立到 CE 插件的 TCP 连接。

    若 _sock 为 None 则新建连接；连接失败返回 False 并记录原因。
    """
    global _sock, _connected
    if _sock is not None:
        # 假定已连接；真实断线会在 send/recv 时报错并触发重连
        return True
    try:
        s = socket.create_connection((HOST, PORT), timeout=SOCKET_TIMEOUT)
        s.settimeout(SOCKET_TIMEOUT)
        _sock = s
        _connected = True
        return True
    except Exception:
        _close_socket()
        return False


def _send_command(cmd_line: str) -> str:
    """向 CE 插件发送一条命令并返回其文本响应。

    流程：
    1. 加锁串行化。
    2. 确保连接；失败返回中文错误。
    3. 发送 ``cmd_line + "\\n"``。
    4. 循环 recv 累积 buffer，直到含 ``\\n`` 或超时/出错；取第一个 ``\\n``
       前的内容作为响应（去掉行尾 ``\\r``）。
    5. 响应为空/超时/断线 -> 关闭 socket（下次请求重连），返回中文错误提示。
    """
    global _sock
    with _lock:
        if not _ensure_connected() or _sock is None:
            return (
                f"无法连接到 CE 插件 ({HOST}:{PORT})，"
                "请确认已启动 Cheat Engine 并加载 CE-MCP-Plugin 插件"
            )

        sock = _sock

        # 发送命令
        try:
            sock.sendall((cmd_line + "\n").encode("utf-8"))
        except Exception as e:
            _sock.sendall((cmd_line + "\n").encode("utf-8"))
        except Exception as e:
            _close_socket()
            return f"发送命令到 CE 插件失败: {e}"

        # 接收响应（累积到出现换行）
        buf = b""
        recv_error = None
        try:
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    # 对端关闭连接
                    break
                buf += chunk
        except socket.timeout:
            # 接收超时：未等到完整的行
            recv_error = "接收超时"
        except Exception as e:
            recv_error = f"接收失败: {e}"

        if b"\n" in buf:
            line = buf.split(b"\n", 1)[0]
            text = line.decode("utf-8", errors="replace").rstrip("\r")
            if text:
                return text
            # 拿到空行
            _close_socket()
            return "CE 插件返回了空响应"
        else:
            # 超时 / 断线 / 未含换行：关闭并重连
            partial = buf.decode("utf-8", errors="replace")
            _close_socket()
            detail = f"，已收到的内容: {partial!r}" if partial else ""
            reason = recv_error or "连接中断"
            return f"未收到 CE 插件完整响应（{reason}{detail}）"


# ---------------------------------------------------------------------------
# 命令调用组装
# ---------------------------------------------------------------------------

def _build_command_line(command_name: str, kwargs: dict) -> str:
    """按参数顺序拼装发送给插件的命令文本。

    - 命令名 + (参数非空时 ":" + ",".join(按顺序存在的参数值))。
    - 参数值为 None 或空字符串的跳过（不占位）。
    """
    values = [str(kwargs[k]).strip() for k in kwargs if str(kwargs[k] or "").strip()]
    if not values:
        return command_name
    return command_name + ":" + ",".join(values)


def _invoke_command(command_name: str, kwargs: dict) -> str:
    """执行单个命令并返回插件响应（始终返回字符串，不抛异常）。"""
    cmd_line = _build_command_line(command_name, kwargs)
    try:
        return _send_command(cmd_line)
    except Exception as e:
        return f"执行命令 {command_name} 时发生错误: {e}"


# ---------------------------------------------------------------------------
# 动态生成 MCP handler
# ---------------------------------------------------------------------------

def _make_handler(command_name: str, params: list, description: str):
    """为一个命令生成带具名可选参数的 handler 函数。

    参数全部为 ``Annotated[Optional[str], Field(description=...)] = None``，
    顺序与解析出的参数名一致；无参数的命令 handler 不接收任何参数。

    返回的 handler 以插件响应字符串作为返回值；失败时返回中文错误字符串。
    """
    if params:
        # 构建具名参数定义，如：
        # def _handler(address: Annotated[Optional[str], Field(...)] = None, ...):
        param_defs = ", ".join(
            f'{p}: Annotated[Optional[str], Field(description="参数: {p}")] = None'
            for p in params
        )
        call_args = ", ".join(f'"{p}": {p}' for p in params)
        body = (
            f"def _handler({param_defs}):\n"
            f'    """{description}"""\n'
            f"    return _invoke_command({command_name!r}, {{{call_args}}})\n"
        )
    else:
        # 无参数命令：handler 不接收参数，直接发送 "COMMAND\n"
        body = (
            "def _handler():\n"
            f'    """{description}"""\n'
            f"    return _invoke_command({command_name!r}, {{}})\n"
        )

    ns = {
        "Annotated": Annotated,
        "Optional": Optional,
        "Field": Field,
        "_invoke_command": _invoke_command,
    }
    exec(compile(body, "<mcp_handler>", "exec"), ns)
    handler = ns["_handler"]
    # 命名与文档
    handler.__name__ = command_name
    handler.__doc__ = description
    return handler


# ---------------------------------------------------------------------------
# 注册 MCP 服务器与 tools
# ---------------------------------------------------------------------------

mcp = MCPServer("CE-MCP-Plugin Bridge")

_registered = 0
for _cmd in _COMMANDS:
    _cmd_name = _cmd.get("name", "")
    if not _cmd_name:
        continue
    _params = _parse_params(_cmd.get("syntax", ""))
    _desc = _cmd.get("description", _cmd_name)
    _handler = _make_handler(_cmd_name, _params, _desc)
    mcp.add_tool(_handler, name=_cmd_name, description=_desc)
    _registered += 1


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Streamable HTTP 传输，监听 127.0.0.1:8080
    # 客户端连接 URL 为 http://127.0.0.1:8080/mcp
    mcp.run(transport="streamable-http", host=BRIDGE_HOST, port=BRIDGE_PORT)
