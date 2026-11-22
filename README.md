# pairwise-roi-ctu-align

## 领域

游戏、图形与多媒体

## 主题

ROI CTU 对齐

## 说明

CTU 对齐、掩码平滑与 QP 偏移。

初始状态下 `media/roi_ctu_align.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
