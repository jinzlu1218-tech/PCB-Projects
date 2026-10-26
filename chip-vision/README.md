# Chip Vision：芯片位姿估计与表面字符识别

本项目完成机器视觉期末小作业的两个功能：

1. **提取芯片位置与角度**：输出芯片本体中心、长宽、角度、四角点、mask 和 bbox，单位均为原始图像像素。
2. **识别芯片表面浅灰激光字符**：输出每行文字、置信度、文字框的四点像素坐标，并把裁正后的芯片图保存下来便于核查。

技术栈为 **Python + OpenCV + PyTorch**。位姿以 SIFT 参考匹配为主，传统暗区域检测为 fallback；字符识别使用 PyTorch 后端的 EasyOCR，并针对本数据集的固定芯片打字做了保守的词表校正。

## 数据与结果

- 输入：`input/board-01.png` ~ `board-20.png`，共 20 张，646×492。
- 芯片：SANYO `LC3664NML-10`，表面文字为：

```text
SANYO
LC3664NML-10
JAPAN
2CB3
```

当前结果：

| 项目 | 结果 |
|---|---|
| 位姿定位成功 | 20 / 20 |
| 位姿置信度范围 | 0.992 ~ 1.000 |
| OCR 完整文本与芯片打字一致 | 20 / 20 |
| SIFT inliers 范围 | 52 ~ 391 |
| 单元测试 | 15 passed |
| 可选 U-Net（伪标签、25 epochs） | 验证集 IoU 0.785 |

说明：

- 位姿没有独立的四角点人工真值，采用 20 张 overlay 逐张人工复核；绿色轮廓贴合芯片本体。
- OCR 的 `text` 是词表校正后的结果，`raw_text` 保留 EasyOCR 原始输出，`corrected` 标记是否发生校正。
- 词表校正是针对本作业固定芯片的可选后处理；若换芯片，应在配置中替换词表或关闭校正。

结果文件：

- `outputs/predictions.json`：完整结构化结果
- `outputs/predictions.csv`：逐图汇总表
- `outputs/overlays/`：芯片框、中心、角度、文字框可视化
- `outputs/masks/`：芯片本体 mask
- `outputs/canonical/`：透视矫正后的芯片图
- `outputs/ocr_input/`：实际送入 OCR 的增强图
- `docs/results/contact_sheet.jpg`：20 张 overlay 总览

## 方法

### 1. 芯片位姿

所有图片来自同一块 PCB、同一颗芯片，因此采用“参考图 + SIFT + 单应性”的高精度方案：

1. 以 `board-01` 为参考图，截取芯片附近模板 `[120, 145, 500, 345]`。
2. 在参考图上标定芯片本体四角。
3. 对模板和目标图提取 SIFT 特征，使用 BFMatcher + Lowe ratio test 筛选匹配。
4. 用 RANSAC 估计单应性矩阵，把参考四角映射到目标图。
5. 做几何校验：
   - inliers 数量与比例；
   - 面积比、长短边比；
   - 四边形方向（拒绝镜像）；
   - 角点是否大部分落在图像内。
6. 校验失败时使用暗区域 + 轮廓 + `minAreaRect` 的 OpenCV fallback。

参考方案对任意平面内旋转都有效。20 张图中旋转接近 ±90°、接近 180° 的样本也能稳定定位。

### 2. 可选 U-Net

`src/chipvision/chip/unet.py` 提供一个小型 U-Net，`scripts/train_unet.py` 使用 SIFT 生成的 pseudo-mask 训练。它是 PyTorch 学习式分割分支，但**默认推理不使用 U-Net**：本数据只有 20 张，参考匹配的几何精度更高。U-Net 主要用于课程中的深度学习要求、或换板后的学习式 fallback。

### 3. 透视矫正

用芯片四角把本体映射到固定 `360×160` 画布。这样任意旋转的芯片都会恢复成水平方向，文字行位置也基本固定。

### 4. 字符识别

1. 在矫正后的芯片图上按相对纵向位置切成三行：
   - SANYO
   - LC3664NML-10
   - JAPAN / 2CB3
2. 每行单独做 CLAHE、反锐化、反色、4 倍放大和白色边框。
3. 每行单独送入 EasyOCR（PyTorch 后端），避免文字检测器把相邻行合并。
4. 把 OCR 四点框按 `H_canonical_to_image` 映射回原图。
5. 用配置中的词表做保守校正：
   - 整行编辑距离不超过阈值；
   - 对 `LC...` 型号行支持从 `LC` 锚点开始匹配，允许一个额外错误；
   - 保留原始文本，绝不无依据地改写。

### 5. 角度约定

输出两个角度：

```text
angle_deg       长边角度，范围 [-90, 90)，逆时针为正
angle_axis_deg  长边轴向角，范围 [0, 180)
```

角度在常规数学坐标系（y 向上）中计算；图像坐标 y 向下，因此代码内部对 `atan2` 结果取了负号。`corners_px` 的顺序固定为参考图的 `p0, p1, p2, p3`，即参考左上、右上、右下、左下。这个顺序随芯片旋转，不能理解为图像坐标中的“左上”。

## 安装

Windows PowerShell：

```powershell
cd "C:\ALL profile undergraduate\major courses\Machine vision\期末小作业"
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

如果使用 GPU，可按照本机 CUDA 版本安装对应 PyTorch，并把 `configs/default.yaml` 中 `ocr.use_gpu` 改为 `true`。

第一次运行 OCR 时 EasyOCR 会下载检测和识别模型到 `weights/easyocr/`。模型文件较大，已在 `.gitignore` 中排除。

## 运行

```powershell
# 完整流程
python -m chipvision infer --input input --output outputs

# 只做位姿，不做 OCR
python -m chipvision infer --no-ocr

# 只跑前 2 张
python -m chipvision infer --limit 2
```

终端会输出：

```text
Done: 20/20 images ok, 0 errors. Results: .../outputs/predictions.json
```

## 输出格式

`outputs/predictions.json`：

```json
{
  "schema_version": "1.0",
  "images": [
    {
      "image": "board-01.png",
      "status": "ok",
      "image_size": [646, 492],
      "chip": {
        "center_px": [308.002, 245.249],
        "angle_deg": -0.085,
        "angle_axis_deg": 179.915,
        "size_px": {"long": 334.007, "short": 136.497},
        "corners_px": [[141.002, 176.004], [475.0, 177.997], [475.006, 312.995], [140.998, 314.0]],
        "bbox_px": [140.0, 176.0, 476.0, 315.0],
        "confidence": 1.0,
        "method": "sift_reference_homography"
      },
      "text": {
        "full_text": "SANYO\nLC3664NML-10\nJAPAN\n2CB3",
        "lines": [
          {
            "text": "LC3664NML-10",
            "raw_text": "LC3864NML -10",
            "corrected": true,
            "confidence": 0.6635,
            "polygon_px": [[183.678, 232.515], [399.538, 233.01], [399.539, 263.729], [183.677, 263.672]]
          }
        ]
      }
    }
  ]
}
```

`outputs/predictions.csv` 包含：

```text
image, status, error, center_x_px, center_y_px, angle_deg, angle_axis_deg,
long_px, short_px, pose_confidence, method, full_text, ocr_confidence
```

## 评估

`chipvision evaluate` 支持按以下格式提供人工标注：

```json
{
  "images": [
    {
      "image": "board-01.png",
      "chip": {"corners_px": [[141, 176], [475, 178], [475, 313], [141, 314]]},
      "text": {"full_text": "SANYO\nLC3664NML-10\nJAPAN\n2CB3"}
    }
  ]
}
```

运行：

```powershell
python -m chipvision evaluate `
  --predictions outputs/predictions.json `
  --annotations data/annotations/ground_truth.json `
  --output outputs/metrics.json
```

指标包括中心点误差、角度误差、四角点 RMSE、字符错误率 CER 和整行完全匹配率。本仓库没有附带人工四角点真值，因此不把预测结果反过来当作 GT 报指标。

## 训练可选 U-Net

```powershell
# 1. 用 SIFT 结果生成伪标签
python scripts/make_pseudo_labels.py

# 2. 训练小 U-Net
python scripts/train_unet.py --epochs 40 --size 384 --batch-size 2

# 3. 查看训练历史
Get-Content weights/chip_unet.history.json
```

本仓库一次 25 epochs、384×384、批大小 2 的参考训练结果为 **验证集 IoU 0.785**。该数字基于 SIFT 伪标签，只说明学习分支可训练，不等同于真实标注精度。

U-Net 默认不参与推理；如需用于新板卡，应采集真实标注并接入 pipeline 的低置信度分支，而不要直接复用本数据集的伪标签。

## 测试与代码检查

```powershell
python -m pytest -q
ruff check src scripts tests
```

测试覆盖：

- 角度归一化与旋转方向；
- 四边形面积、mask；
- 参考图自匹配；
- 20 张图全部定位；
- Unicode 路径读写；
- 透视矫正与 OCR 预处理；
- 词表纠错；
- Pipeline pose-only 端到端。

## 目录结构

```text
.
├─ configs/default.yaml          # 参考图、匹配参数、OCR 参数
├─ input/                        # 20 张原始图片 + readme
├─ src/chipvision/
│  ├─ chip/
│  │  ├─ matcher.py              # SIFT 参考匹配（主方案）
│  │  ├─ classical.py            # 暗区域轮廓 fallback
│  │  ├─ unet.py                 # 可选 PyTorch U-Net
│  │  └─ types.py                # PoseResult
│  ├─ text/
│  │  ├─ rectify.py              # 四角透视矫正
│  │  ├─ preprocess.py           # CLAHE/反色/放大
│  │  ├─ recognizer.py           # 分行 OCR + 坐标映射
│  │  └─ lexicon.py              # 保守词表校正
│  ├─ geometry.py                # 角度、角点、mask
│  ├─ io.py                      # Unicode 安全图片读写
│  ├─ pipeline.py                # 批处理与结果落盘
│  ├─ visualize.py
│  ├─ evaluate.py
│  └─ cli.py
├─ scripts/
│  ├─ make_pseudo_labels.py
│  ├─ train_unet.py
│  └─ make_contact_sheet.py
├─ tests/
├─ docs/results/contact_sheet.jpg
└─ outputs/
```

## 复现说明

- 参考图为 `input/board-01.png`，参考四角写在 `configs/default.yaml`。
- SIFT 参数、RANSAC 阈值、OCR 行带和词表也在同一配置文件。
- 运行结果确定性强；EasyOCR 首次运行需要联网下载模型。
- 原始图片保留在仓库中，约 4 MB，便于复现。
- `.venv/`、`data/pseudo/`、`weights/`、OCR 中间图不提交。

## 限制

1. **参考匹配依赖同一块板、同一颗芯片**。换成新板卡时 SIFT 模板会失效，需要重新标定参考图或使用学习式模型。
2. 词表校正是针对本数据集固定芯片的可选步骤，不应直接用于任意芯片型号。
3. 没有独立的人工四角点 GT，位姿精度以 overlay 人工复核为主。
4. 极小、低对比度的激光字在部分图像上仍可能误读；本项目通过分行 OCR + 词表校正解决本数据集的问题，但没有声称通用 OCR 已达到工业级。
5. U-Net 使用伪标签训练，只适合作为课程中的 PyTorch 分支和新板卡实验起点。

## License

MIT，见 `LICENSE`。

GitHub 仓库：<https://github.com/jinzlu1218-tech/PCB-Projects>
