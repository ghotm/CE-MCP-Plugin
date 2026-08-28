"""
CE-MCP-Plugin 桥接服务器（Streamable HTTP MCP Server）
========================================================

本模块实现一个 MCP（Model Context Protocol）桥接服务器，用于把 Cheat Engine
插件 CE-MCP-Plugin 暴露的 75 个（配置实际为 77 个）命令封装成 MCP tools，
供支持 MCP 的 AI 客户端调用。

协议概述
--------
- CE-MCP-Plugin 插件是 TCP **客户端**，启动后自动连接本机 127.0.0.1:8888
  并无限重试（见 /workspace/CE-MCP-Plugin/mcp.json 中的 mcp.server 字段）。
- 本桥在 127.0.0.1:8888 扮演 TCP **服务器**（即原作者设计中"AI 服务器"的
  角色），接受插件的连接，避免"双方都是客户端"造成的死锁。
- 命令通过 TCP 以文本行方式交互：
  * 发送：一行文本命令，以 ``\\n`` 结尾，形如 ``COMMAND:param1,param2`` 或
    无参数命令 ``COMMAND``。
  * 接收：插件执行后回传一行结果文本，以 ``\\n`` 结尾。
- 本桥同时是一个 Streamable HTTP MCP 服务器，监听 127.0.0.1:8080，MCP 客户端
  连接 URL 为 http://127.0.0.1:8080/mcp。

职责
----
1. 读取 mcp.json 配置，收集全部命令（名称、中文描述、语法）。
2. 从命令语法（syntax）解析参数名。
3. 为每个命令动态注册一个 MCP tool。
4. 通过 TCP 服务器连接把 MCP 调用转发给 CE 插件，并返回插件的文本响应。

依赖
----
- Python >= 3.10
- 标准库 + 官方 ``mcp`` 包（``from mcp.server import MCPServer``）。
"""

import json
import socket
import threading
import time
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

# CE 插件要连接的地址与端口（本桥在此监听，扮演插件所连接的"AI 服务器"）
HOST = _CONFIG.get("mcp", {}).get("server", {}).get("host", "127.0.0.1")
PORT = _CONFIG.get("mcp", {}).get("server", {}).get("port", 8888)

# 接收响应超时（秒）
SOCKET_TIMEOUT = 10.0

# 非阻塞 accept 的短超时（秒）：没有插件连接时快速返回，不阻塞 MCP 调用
ACCEPT_TIMEOUT = 0.1

# 桥的 MCP 服务器监听地址与端口
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
# TCP 服务器（等待 CE 插件连接）
# ---------------------------------------------------------------------------

# 服务器监听 socket（模块级，只创建一次）
_server_sock: Optional[socket.socket] = None
# 当前插件连接 socket（可能为 None）
_plugin_sock: Optional[socket.socket] = None
# 串行化所有请求：插件同时只接受一个连接、一次处理一个命令
_lock = threading.Lock()

try:
    _server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 允许端口快速复用，避免重启时 TIME_WAIT 占用
    _server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _server_sock.bind((HOST, PORT))
    _server_sock.listen(1)
except Exception as e:
    print(f"[桥] 错误：无法在 {HOST}:{PORT} 创建监听 socket：{e}")
    _server_sock = None


def _close_plugin_socket() -> None:
    """关闭并清空当前插件连接 socket。"""
    global _plugin_sock
    if _plugin_sock is not None:
        try:
            _plugin_sock.close()
        except Exception:
            pass
    _plugin_sock = None


def _ensure_plugin_connected() -> bool:
    """确保已接受到 CE 插件的连接（带死连接过滤）。

    - 若当前插件连接存在则视为有效，直接返回 True。
    - 否则在短超时窗口内循环 accept，并对每个新连接先用 MSG_PEEK 探测活性
      （peek 不消费数据）：
      * 抛 BlockingIOError（对端活着但暂无数据）或 recv 到数据 -> 活连接，
        恢复超时模式并采用为新连接，返回 True；
      * recv 返回 b""（对端已关闭/EOF）或抛 ConnectionResetError/OSError
        -> 死连接，close() 丢弃并 continue 继续 accept 下一个；
      * accept 超时（socket.timeout）-> 返回 False（无连接，不阻塞等待，
        MCP 调用应立即返回错误而非挂起）。
    """
    global _plugin_sock
    if _plugin_sock is not None:
        # 假定连接有效；真实断线会在 send/recv 时报错并触发下次重连
        return True
    if _server_sock is None:
        return False
    # 总等待时长：留出依次丢弃多个死连接的时间，避免整体阻塞
    deadline = time.monotonic() + ACCEPT_TIMEOUT
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # 等待窗口耗尽，仍未等到活连接
            return False
        _server_sock.settimeout(remaining)
        try:
            conn, _addr = _server_sock.accept()
        except Exception:
            # accept 超时（socket.timeout，短超时内无插件连上来）或其他异常
            # （服务器 socket 出错等）均视为无连接，不阻塞等待，立即返回
            return False

        # 验证连接活性：MSG_PEEK 探测不消费数据
        alive = False
        try:
            conn.setblocking(False)
            if conn.recv(1, socket.MSG_PEEK) != b"":
                # 对端已有数据可读（主动发来内容）-> 活连接
                alive = True
        except Exception as e:
            if isinstance(e, BlockingIOError):
                # 对端活着但暂无数据 -> 活连接
                alive = True
            else:
                # 连接被重置/出错 -> 死连接
                alive = False

        if not alive:
            # 死连接：close() 丢弃，继续 accept 下一个
            try:
                conn.close()
            except Exception:
                pass
            continue

        # 活连接：恢复接收超时模式并采用
        conn.settimeout(SOCKET_TIMEOUT)
        _plugin_sock = conn
        print(f"CE 插件已连接 ({HOST}:{PORT})")
        return True


def _send_command(cmd_line: str) -> str:
    """向 CE 插件发送一条命令并返回其文本响应（TCP 服务器模式）。

    流程：
    1. 加锁串行化。
    2. 确保插件连接；未连接则返回中文错误（不阻塞，accept 最多等 0.1 秒）。
    3. 发送 ``cmd_line + "\\n"``。
    4. 循环 recv 累积 buffer，直到含 ``\\n`` 或超时/出错；取第一个 ``\\n``
       前的内容作为响应（去掉行尾 ``\\r``）。
    5. 写或读失败 -> 关闭当前插件连接（下次调用会重新 accept），返回中文错误。
    """
    global _plugin_sock
    with _lock:
        if not _ensure_plugin_connected() or _plugin_sock is None:
            return (
                f"无法连接 CE 插件：桥正在 {HOST}:{PORT} 监听，但插件尚未连上。"
                "请确认：1) 已在 Cheat Engine 中加载 CE-MCP-Plugin 插件；"
                "2) 桥先于插件启动（插件会自动重连）。"
            )

        conn = _plugin_sock

        # 发送命令
        try:
            conn.sendall((cmd_line + "\n").encode("utf-8"))
        except Exception:
            _close_plugin_socket()
            return "与 CE 插件的连接已断开，请检查插件状态（插件会自动重连）"

        # 接收响应（累积到出现换行）
        buf = b""
        recv_error = None
        try:
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    # 对端关闭连接
                    break
                buf += chunk
        except Exception as e:
            if isinstance(e, socket.timeout):
                # 接收超时：未等到完整的行
                recv_error = "接收超时"
            else:
                recv_error = f"接收失败: {e}"

        if b"\n" in buf:
            line = buf.split(b"\n", 1)[0]
            text = line.decode("utf-8", errors="replace").rstrip("\r")
            if text:
                return text
            # 拿到空行
            _close_plugin_socket()
            return "CE 插件返回了空响应（插件会自动重连）"
        else:
            # 超时 / 断线 / 未含换行：关闭当前连接，下次调用重新 accept
            partial = buf.decode("utf-8", errors="replace")
            _close_plugin_socket()
            detail = f"，已收到的内容: {partial!r}" if partial else ""
            reason = recv_error or "连接中断"
            return f"与 CE 插件的连接已断开，请检查插件状态（插件会自动重连）（{reason}{detail}）"


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
    # 打印启动日志：桥在 TCP 8888 端扮演服务器，等待插件连接
    print(f"MCP 桥已启动，等待 CE 插件连接 {HOST}:{PORT} ...")
    # Streamable HTTP 传输，监听 127.0.0.1:8080
    # MCP 客户端连接 URL 为 http://127.0.0.1:8080/mcp
    mcp.run(transport="streamable-http", host=BRIDGE_HOST, port=BRIDGE_PORT)