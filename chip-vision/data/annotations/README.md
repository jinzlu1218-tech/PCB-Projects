# 人工标注格式

本项目没有附带人工四角点 GT，避免把预测结果反过来当作真值。需要计算位姿指标时，按下面格式建立 `ground_truth.json`：

```json
{
  "images": [
    {
      "image": "board-01.png",
      "chip": {
        "corners_px": [[141, 176], [475, 178], [475, 313], [141, 314]]
      },
      "text": {
        "full_text": "SANYO\nLC3664NML-10\nJAPAN\n2CB3"
      }
    }
  ]
}
```

四角点顺序必须与项目约定一致：

```text
p0 参考左上
p1 参考右上
p2 参考右下
p3 参考左下
```

运行评估：

```powershell
python -m chipvision evaluate `
  --predictions outputs/predictions.json `
  --annotations data/annotations/ground_truth.json `
  --output outputs/metrics.json
```
