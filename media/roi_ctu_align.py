"""ROI / CTU 对齐工具集。

用于游戏与多媒体编码流水线中，把逐帧重算出来的 ROI（感兴趣区域）
与编码器的 CTU（编码树单元）网格对齐，并对 ROI 掩码做平滑、按区域
给出 QP 偏移，从而避免“边界对不齐导致码率不降反升、边缘块效应明显”。

模块包含：

* CTU 网格与区域对齐： ``ctu_grid`` / ``ctu_align`` / ``edge_guard``
* ROI 掩码处理： ``mask_dilate`` / ``roi_mask_smooth``
* QP 控制： ``qp_offset`` / ``fallback_qp`` / ``rate_control``
* 边界与块效应： ``boundary_block`` / ``block_artifact``
* 码率度量： ``bitrate_compare`` / ``bitrate_metric``
* 其他： ``roi_recompute`` / ``validation`` / ``batch_guard``

所有入口都做防御式处理：非法输入不会抛异常，而是退化为合理默认值，
且返回值始终非 ``None``。
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

__all__ = [
    "ctu_grid",
    "ctu_align",
    "roi_mask_smooth",
    "qp_offset",
    "boundary_block",
    "bitrate_compare",
    "roi_recompute",
    "block_artifact",
    "validation",
    "mask_dilate",
    "rate_control",
    "fallback_qp",
    "bitrate_metric",
    "batch_guard",
    "edge_guard",
]

DEFAULT_CTU = 64
DEFAULT_QP = 26
MIN_QP = 0
MAX_QP = 51
DEFAULT_QSTEP = 6.0
DEFAULT_ROI_STRENGTH = 3.0


# ---------------------------------------------------------------------------
# 输入规整辅助
# ---------------------------------------------------------------------------

def _is_seq(value):
    """判断是否为序列（字符串/字节串不算，避免被逐字符遍历）。"""
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _is_matrix(value):
    """判断是否为二维序列。"""
    if not _is_seq(value) or len(value) == 0:
        return False
    return any(_is_seq(row) for row in value)


def _num(value, default=0.0):
    """尽量把输入转成有限的 float；失败或非有限值返回 default。"""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _floats(seq):
    return [_num(item, 0.0) for item in seq]


def _clamp(value, low, high):
    return low if value < low else high if value > high else value


def _positive_int(value, default):
    try:
        result = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return int(default)
    return result if result > 0 else int(default)


def _as_int(value, default=0):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _clamp_qp(value):
    return _clamp(_as_int(value, DEFAULT_QP), MIN_QP, MAX_QP)


def _floor_to(value, step):
    return int(math.floor(value / step) * step)


def _ceil_to(value, step):
    return int(math.ceil(value / step) * step)


def _misalignment(value, ctu):
    """坐标到最近 CTU 边界的距离，归一化到 0..0.5。"""
    if ctu <= 0:
        return 0.0
    remainder = value % ctu
    if remainder == 0:
        return 0.0
    return min(remainder, ctu - remainder) / ctu


def _guarded(fallback):
    """捕获内部异常，保证对外永不抛出（fallback 必须非 None）。"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:  # pragma: no cover - 纯防御分支
                if callable(fallback):
                    try:
                        return fallback()
                    except Exception:
                        return 0
                return fallback

        return wrapper

    return decorator


def _mask_like(values, original):
    """把 float 结果还原成更贴近输入的类型（整型/布尔掩码保持整型）。"""
    reference = []
    if _is_matrix(original):
        for row in original:
            if _is_seq(row):
                reference.extend(row)
    elif _is_seq(original):
        reference = list(original)

    if reference and all(
        isinstance(v, bool) or (isinstance(v, int) and not isinstance(v, bool))
        for v in reference
    ):
        return [int(round(v)) for v in values]
    return [round(v, 6) for v in values]


# ---------------------------------------------------------------------------
# CTU 网格与对齐
# ---------------------------------------------------------------------------

@_guarded(dict)
def ctu_grid(width, height=None, ctu=DEFAULT_CTU):
    """构建 CTU 网格描述。

    支持 ``ctu_grid(width, height, ctu)`` 以及 ``ctu_grid((width, height), ctu)``。
    返回包含 CTU 尺寸、行列数与总数的字典。
    """
    if _is_seq(width):
        values = _floats(width)
        w = values[0] if len(values) > 0 else 0.0
        h = values[1] if len(values) > 1 else w
        if len(values) > 2:
            ctu = values[2]
        elif height is not None:
            ctu = height
    else:
        w = _num(width, 0.0)
        h = w if height is None else _num(height, 0.0)

    ctu_size = _positive_int(ctu, DEFAULT_CTU)
    columns = max(1, int(math.ceil(max(0.0, w) / ctu_size)))
    rows = max(1, int(math.ceil(max(0.0, h) / ctu_size)))
    return {
        "ctu": ctu_size,
        "ctu_size": ctu_size,
        "width": w,
        "height": h,
        "columns": columns,
        "rows": rows,
        "cols": columns,
        "count": columns * rows,
        "ctu_width": ctu_size,
        "ctu_height": ctu_size,
    }


@_guarded(0)
def ctu_align(region, ctu=DEFAULT_CTU, mode="expand"):
    """把 ROI 区域/尺寸对齐到 CTU 边界。

    * 长度为 4 的序列按 ``(x0, y0, x1, y1)`` 解释，``mode="expand"`` 时向外
      扩展，``"shrink"`` 时向内收缩，``"nearest"`` 时取最近边界。
    * 长度为 2 的序列按 ``(width, height)`` 解释，返回对齐后的尺寸元组。
    * 单个数值返回向上对齐后的长度（例如 ``ctu_align(100, 64) == 128``）。

    返回区域使用与输入一致的“左上-右下”坐标顺序。
    """
    ctu_size = _positive_int(ctu, DEFAULT_CTU)
    mode = mode if isinstance(mode, str) else "expand"

    if _is_seq(region):
        values = _floats(region)
        if len(values) >= 4:
            x0, y0, x1, y1 = values[:4]
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            if mode == "shrink":
                ax0, ay0 = _ceil_to(x0, ctu_size), _ceil_to(y0, ctu_size)
                ax1, ay1 = _floor_to(x1, ctu_size), _floor_to(y1, ctu_size)
                # 区域小于一个 CTU 时退化为包含起点的单个 CTU
                if ax1 <= ax0:
                    ax0 = _floor_to(x0, ctu_size)
                    ax1 = ax0 + ctu_size
                if ay1 <= ay0:
                    ay0 = _floor_to(y0, ctu_size)
                    ay1 = ay0 + ctu_size
            elif mode == "nearest":
                ax0 = int(round(x0 / ctu_size) * ctu_size)
                ay0 = int(round(y0 / ctu_size) * ctu_size)
                ax1 = int(round(x1 / ctu_size) * ctu_size)
                ay1 = int(round(y1 / ctu_size) * ctu_size)
            else:  # expand
                ax0, ay0 = _floor_to(x0, ctu_size), _floor_to(y0, ctu_size)
                ax1, ay1 = _ceil_to(x1, ctu_size), _ceil_to(y1, ctu_size)

            return (
                max(0, ax0),
                max(0, ay0),
                max(0, ax1),
                max(0, ay1),
            )
        if len(values) == 3:
            x0, y0, size = values
            return ctu_align((x0, y0, x0 + size, y0 + size), ctu_size, mode)
        if len(values) == 2:
            return (_ceil_to(values[0], ctu_size), _ceil_to(values[1], ctu_size))
        if len(values) == 1:
            return _ceil_to(values[0], ctu_size)
        return (0, 0, 0, 0)

    return _ceil_to(_num(region, 0.0), ctu_size)


@_guarded(0)
def edge_guard(region, width=None, height=None):
    """把区域裁剪到帧边界内，避免越界访问。

    * 标量：裁剪到 ``[0, width]``。
    * 4 元组 ``(x0, y0, x1, y1)``：裁剪到 ``width`` / ``height`` 范围内，
      并保证 ``x0 <= x1``、``y0 <= y1``。
    * 2 元组：按尺寸裁剪（超过帧尺寸时截断）。
    """
    if not _is_seq(region):
        value = _num(region, 0.0)
        if width is None:
            return _as_int(value) if value.is_integer() else value
        limit = max(0.0, _num(width, 0.0))
        result = _clamp(value, 0.0, limit)
        return _as_int(result) if result.is_integer() else result

    values = _floats(region)
    if len(values) >= 4:
        x0, y0, x1, y1 = values[:4]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        max_w = math.inf if width is None else max(0.0, _num(width, 0.0))
        max_h = (
            max_w if height is None else max(0.0, _num(height, 0.0))
        )
        x0 = _clamp(x0, 0.0, max_w)
        x1 = _clamp(x1, 0.0, max_w)
        y0 = _clamp(y0, 0.0, max_h)
        y1 = _clamp(y1, 0.0, max_h)
        return tuple(
            _as_int(v) if float(v).is_integer() else v for v in (x0, y0, x1, y1)
        )
    if len(values) == 2:
        max_w = math.inf if width is None else max(0.0, _num(width, 0.0))
        max_h = (
            max_w if height is None else max(0.0, _num(height, 0.0))
        )
        w = _clamp(values[0], 0.0, max_w)
        h = _clamp(values[1], 0.0, max_h)
        return (_as_int(w), _as_int(h))
    if len(values) == 1:
        return edge_guard(values[0], width, height)
    return tuple()


# ---------------------------------------------------------------------------
# 掩码处理
# ---------------------------------------------------------------------------

def _dilate_1d(values, radius):
    count = len(values)
    out = []
    for index in range(count):
        start = max(0, index - radius)
        end = min(count, index + radius + 1)
        out.append(max(values[start:end]))
    return out


def _dilate_2d(grid, radius):
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    padded = [
        [_num(row[col], 0.0) if col < len(row) else 0.0 for col in range(cols)]
        for row in grid
    ]
    out = [[0.0] * cols for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            best = -math.inf
            for ny in range(max(0, y - radius), min(rows, y + radius + 1)):
                for nx in range(max(0, x - radius), min(cols, x + radius + 1)):
                    best = max(best, padded[ny][nx])
            out[y][x] = best
    return out


@_guarded(list)
def mask_dilate(mask, radius=1):
    """对 ROI 掩码做最大值膨胀，扩大 ROI 以覆盖 CTU 对齐后的边缘。

    支持标量、一维掩码与二维掩码。标量输入原样返回（非负数值）。
    """
    radius = max(0, _as_int(radius, 1))
    if not _is_seq(mask):
        return max(0.0, _num(mask, 0.0)) if not isinstance(mask, int) else max(0, mask)

    if _is_matrix(mask):
        pairs = [(_floats(row), row) for row in mask if _is_seq(row)]
        rows = [values for values, _ in pairs]
        if radius == 0:
            return [_mask_like(values, original) for values, original in pairs]
        dilated = _dilate_2d(rows, radius)
        return [
            _mask_like(values, original)
            for values, (_, original) in zip(dilated, pairs)
        ]

    values = _floats(mask)
    if radius == 0:
        return _mask_like(values, mask)
    return _mask_like(_dilate_1d(values, radius), mask)


def _blur_1d(values, radius):
    count = len(values)
    out = []
    for index in range(count):
        start = max(0, index - radius)
        end = min(count, index + radius + 1)
        window = values[start:end]
        out.append(sum(window) / len(window) if window else 0.0)
    return out


def _blur_2d(grid, radius):
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    padded = [
        [_num(row[col], 0.0) if col < len(row) else 0.0 for col in range(cols)]
        for row in grid
    ]
    horizontal = [_blur_1d(row, radius) for row in padded]
    out = [[0.0] * cols for _ in range(rows)]
    for x in range(cols):
        column = [horizontal[y][x] for y in range(rows)]
        blurred = _blur_1d(column, radius)
        for y in range(rows):
            out[y][x] = blurred[y]
    return out


@_guarded(list)
def roi_mask_smooth(mask, radius=1):
    """对 ROI 掩码做盒式平滑，软化 ROI/非 ROI 边界。

    返回值会被限制在 ``[0, 1]``（若输入是 0/1 掩码），以避免 QP 跳变
    造成的边缘块效应。标量输入按其强度返回 ``[0, 1]`` 内的浮点数。
    """
    radius = max(0, _as_int(radius, 1))
    if not _is_seq(mask):
        return _clamp(_num(mask, 0.0), 0.0, 1.0)

    if _is_matrix(mask):
        rows = [_floats(row) for row in mask if _is_seq(row)]
        if radius == 0:
            return [[_clamp(v, 0.0, 1.0) for v in row] for row in rows]
        blurred = _blur_2d(rows, radius)
        return [[_clamp(v, 0.0, 1.0) for v in row] for row in blurred]

    values = [_clamp(v, 0.0, 1.0) for v in _floats(mask)]
    if radius == 0:
        return [round(v, 6) for v in values]
    return [round(v, 6) for v in _blur_1d(values, radius)]


# ---------------------------------------------------------------------------
# QP 与码控
# ---------------------------------------------------------------------------

@_guarded(0)
def qp_offset(roi, base_qp=DEFAULT_QP, strength=DEFAULT_ROI_STRENGTH,
              max_delta=DEFAULT_QSTEP):
    """按区域重要度给出 QP 偏移。

    ``roi`` 为 ``0..1`` 的权重（也接受 ``0..100`` 的百分比）。ROI 越重要，
    偏移越负（质量越高）；背景偏移为 0。返回值为整数偏移量，可直接加到
    ``base_qp`` 上，并保证结果落在合法 QP 范围内。

    序列输入返回逐个元素的偏移列表。
    """
    if _is_seq(roi):
        return [
            qp_offset(item, base_qp, strength, max_delta) for item in roi
        ]

    importance = _num(roi, 0.0)
    if importance > 1.0:
        importance /= 100.0
    importance = _clamp(importance, 0.0, 1.0)

    strength = max(0.0, _num(strength, DEFAULT_ROI_STRENGTH))
    max_delta = max(0.0, _num(max_delta, DEFAULT_QSTEP))
    base = _clamp_qp(base_qp)

    delta = -min(max_delta, strength * importance)
    delta_int = int(round(delta))
    # 保证 base + delta 仍在 [MIN_QP, MAX_QP] 内
    delta_int = max(delta_int, MIN_QP - base)
    delta_int = min(delta_int, MAX_QP - base)
    return int(delta_int)


@_guarded(0)
def fallback_qp(qp, fallback=DEFAULT_QP):
    """输入非法时回退到安全 QP，并把合法值裁剪进 ``[0, 51]``。"""
    try:
        value = float(qp)
    except (TypeError, ValueError, OverflowError):
        return _clamp_qp(fallback)
    if math.isnan(value) or math.isinf(value):
        return _clamp_qp(fallback)
    return _clamp_qp(value)


@_guarded(0)
def rate_control(target_bits, actual_bits=None, base_qp=DEFAULT_QP,
                 strength=DEFAULT_QSTEP):
    """根据实际码率与目标码率的偏差给出建议 QP。

    实际码率偏高（``actual/target > 1``）时提高 QP 以压低码率，偏低时
    降低 QP 以保护质量；结果限制在 ``[0, 51]``。
    """
    target = _num(target_bits, 0.0)
    base = _clamp_qp(base_qp)
    if actual_bits is None:
        return base
    actual = _num(actual_bits, 0.0)
    if target <= 0.0 or actual <= 0.0:
        return base

    ratio = actual / target
    strength = max(0.0, _num(strength, DEFAULT_QSTEP))
    delta = int(round(strength * math.log2(ratio)))
    return _clamp(base + delta, MIN_QP, MAX_QP)


# ---------------------------------------------------------------------------
# 边界与块效应
# ---------------------------------------------------------------------------

@_guarded(False)
def boundary_block(region, ctu=DEFAULT_CTU):
    """判断区域是否跨越 / 未对齐 CTU 边界。

    4 元组检查四条边是否都落在 CTU 网格线上；标量检查该长度是否为
    CTU 的整数倍；``(width, height)`` 检查两个维度。
    """
    ctu_size = _positive_int(ctu, DEFAULT_CTU)

    if not _is_seq(region):
        return _num(region, 0.0) % ctu_size != 0

    values = _floats(region)
    if len(values) >= 4:
        x0, y0, x1, y1 = values[:4]
        return any(
            coord % ctu_size != 0 for coord in (x0, y0, x1, y1)
        )
    return any(value % ctu_size != 0 for value in values)


@_guarded(0.0)
def block_artifact(region, ctu=DEFAULT_CTU, strength=1.0):
    """估计未对齐导致的块效应强度，返回 ``0..1`` 的分数。

    依据 ROI 边界到最近 CTU 网格线的平均距离：越贴近 CTU 边界，
    边缘块效应越弱。标量按单个尺寸处理。
    """
    ctu_size = _positive_int(ctu, DEFAULT_CTU)
    strength = max(0.0, _num(strength, 1.0))

    if not _is_seq(region):
        score = _misalignment(_num(region, 0.0), ctu_size)
        return _clamp(score * strength, 0.0, 1.0)

    values = _floats(region)
    if len(values) >= 4:
        coords = values[:4]
    else:
        coords = values
    if not coords:
        return 0.0
    score = sum(_misalignment(value, ctu_size) for value in coords) / len(coords)
    return _clamp(score * strength, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 码率度量
# ---------------------------------------------------------------------------

@_guarded(0.0)
def bitrate_compare(baseline, candidate=None, percent=True):
    """比较 ROI 编码前后的码率。

    标量输入返回 ``candidate`` 相对 ``baseline`` 的变化量；``percent=True``
    时返回百分比。序列输入返回逐项结果。``baseline`` 为 0 时返回 0.0。
    """
    scale = 100.0 if percent else 1.0

    if _is_seq(baseline):
        base_values = _floats(baseline)
        if candidate is None:
            return []
        if not _is_seq(candidate):
            candidate = [candidate] * len(base_values)
        cand_values = _floats(candidate)
        count = min(len(base_values), len(cand_values))
        results = []
        for index in range(count):
            base = base_values[index]
            cand = cand_values[index]
            results.append(
                round((cand - base) / base * scale, 6) if base != 0 else 0.0
            )
        return results

    base = _num(baseline, 0.0)
    cand = _num(candidate, 0.0)
    if base == 0.0:
        return 0.0
    return round((cand - base) / base * scale, 6)


@_guarded(0.0)
def bitrate_metric(bits, duration=1.0, unit="kbps"):
    """把比特数 / 时长换算成码率，默认单位为 kbps。"""
    bits = _num(bits, 0.0)
    duration = _num(duration, 1.0)
    if duration <= 0.0:
        duration = 1.0

    factors = {"bps": 1.0, "kbps": 1e3, "mbps": 1e6, "gbps": 1e9}
    key = unit.lower() if isinstance(unit, str) else "kbps"
    factor = factors.get(key, 1e3)
    return round(bits / duration / factor, 6)


# ---------------------------------------------------------------------------
# ROI 重算与守卫
# ---------------------------------------------------------------------------

def _change_ratio(previous, current):
    """计算两个掩码/数值之间的变化程度，归一化到 0..1。"""
    if not _is_seq(previous) or not _is_seq(current):
        prev = _num(previous, 0.0)
        cur = _num(current, 0.0)
        scale = max(abs(prev), abs(cur), 1.0)
        return _clamp(abs(cur - prev) / scale, 0.0, 1.0)

    prev_values = _floats(previous)
    cur_values = _floats(current)
    if len(prev_values) != len(cur_values):
        return 1.0
    if not prev_values:
        return 0.0

    # 二值掩码用 1 - IoU 衡量变化，更适合 ROI 区域比较
    binary = all(v in (0.0, 1.0) for v in prev_values + cur_values)
    if binary:
        union = sum(1 for a, b in zip(prev_values, cur_values) if a or b)
        if union == 0:
            return 0.0
        inter = sum(1 for a, b in zip(prev_values, cur_values) if a and b)
        return _clamp(1.0 - inter / union, 0.0, 1.0)

    total = 0.0
    for a, b in zip(prev_values, cur_values):
        scale = max(abs(a), abs(b), 1.0)
        total += abs(b - a) / scale
    return _clamp(total / len(prev_values), 0.0, 1.0)


@_guarded(False)
def roi_recompute(previous, current=None, threshold=0.1):
    """判断 ROI 是否需要重算。

    当两帧 ROI 掩码的变化比例超过 ``threshold`` 时返回 ``True``。
    仅传入一个参数时，视为逐帧重算请求，直接返回 ``True``。
    """
    if current is None:
        return True
    threshold = _clamp(_num(threshold, 0.1), 0.0, 1.0)
    return _change_ratio(previous, current) >= threshold


@_guarded(dict)
def validation(region=None, ctu=DEFAULT_CTU, **kwargs):
    """校验 ROI 区域、掩码与 CTU 参数，返回结构化报告。

    返回值包含 ``valid`` / ``issues`` / ``warnings``，非法输入不会抛异常。
    """
    issues = []
    warnings = []

    ctu_size = _positive_int(ctu, DEFAULT_CTU)
    if ctu_size != _num(ctu, DEFAULT_CTU):
        warnings.append("ctu 被调整为整数")
    if ctu_size < 4:
        warnings.append("ctu 小于 4，编码器可能不支持")
    if ctu_size & (ctu_size - 1):
        warnings.append("ctu 不是 2 的幂，部分编码器要求 2 的幂")

    report = {
        "valid": True,
        "issues": issues,
        "warnings": warnings,
        "ctu": ctu_size,
        "region": region,
    }

    if region is None:
        warnings.append("未提供 ROI 区域")
        return report

    if _is_matrix(region):
        for row in region:
            if not _is_seq(row):
                continue
            for value in row:
                numeric = _num(value, math.nan)
                if math.isnan(numeric) or numeric < 0.0 or numeric > 1.0:
                    issues.append("掩码取值超出 [0, 1]")
                    break
            if issues:
                break
    elif _is_seq(region):
        values = _floats(region)
        if len(values) not in (2, 3, 4):
            warnings.append("区域维度数量不是 2/3/4")
        if any(value < 0.0 for value in values):
            issues.append("区域包含负坐标")
        if len(values) >= 4:
            if values[2] < values[0] or values[3] < values[1]:
                issues.append("区域右下角小于左上角")
    else:
        try:
            value = float(region)
        except (TypeError, ValueError, OverflowError):
            issues.append("区域不是数值")
        else:
            if math.isnan(value) or math.isinf(value) or value < 0.0:
                issues.append("区域不是有限的非负数值")
            elif value == 0.0:
                warnings.append("区域尺寸为 0")

    for name, value in kwargs.items():
        if value is None:
            warnings.append(f"参数 {name} 为空")
        elif _is_seq(value):
            if any(_num(item, math.nan) != _num(item, math.nan) for item in value):
                issues.append(f"参数 {name} 含非数值")

    report["valid"] = not issues
    return report


@_guarded(list)
def batch_guard(items, limit=None):
    """批量处理守卫：清理非法项，并按 ``limit`` 截断。

    标量输入视为批大小，返回裁剪到 ``[0, limit]`` 的整数。
    序列输入返回过滤掉 ``None`` / 非数值后的新列表。
    """
    limit_value = None
    if limit is not None:
        try:
            limit_value = max(0, int(limit))
        except (TypeError, ValueError, OverflowError):
            limit_value = None

    if not _is_seq(items):
        if items is None:
            return []
        value = _num(items, 0.0)
        if limit_value is not None:
            value = _clamp(value, 0.0, float(limit_value))
        return _as_int(value) if float(value).is_integer() else value

    cleaned = []
    for item in items:
        if item is None:
            continue
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        cleaned.append(item)
        if limit_value is not None and len(cleaned) >= limit_value:
            break
    return cleaned
