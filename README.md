# pairwise-tide-buoyancy

## 领域

游戏、图形与多媒体

## 主题

潮汐水位与浮力采样

## 说明

物理同潮位、查询就绪前冻结与连续浪高。

初始状态下 `gameplay/tide_buoyancy.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
