# AUV-MADRL Showcase

多艇协同与攻防自博弈的个人工程探索展示，由 **lvdacong** 主导搭建仿真、强化学习训练、模型部署和交互可视化流程，用于展示软件实现与系统集成能力。开发过程中使用 AI 编程工具辅助实现和调试，展示视频由作者本人录制；第三方库与素材不属于作者原创。

**目前未发表与本项目相关的论文，项目暂时停滞。** 这些录屏记录的是探索过程中的行为现象，不构成收敛、最优策略、泛化能力或实艇有效性的科研结论。

## 展示：从协同避障到三角色自博弈

以下六段视频按作者确认的递进顺序排列。前五段保留历史命名及当时画面，名称中的策略解释是探索记录，不代表经过独立实验验证的结论。

| 顺序 | 视频 | 时长 |
| --- | --- | --- |
| 1 | [多艇协同避障](docs/videos/01-多艇协同避障.mp4) | 52 秒 |
| 2 | [捕风捉影](docs/videos/02-捕风捉影.mp4) | 9 秒 |
| 3 | [公平攻防自博弈：DEF 更高策略轮次](docs/videos/03-公平攻防自博弈-DEF更高策略轮次.mp4) | 19 秒 |
| 4 | [ATK–DEF 自动阵型排布](docs/videos/04-ATK-DEF-自动阵型排布.mp4) | 26 秒 |
| 5 | [TGT 帮助 DEF 获得学习信号](docs/videos/05-tgt帮助def获得学习信号.mp4) | 23 秒 |
| 6 | **[三角色、双阵营自博弈：当前展示版本](docs/videos/06-三角色双阵营自博弈.mp4)** | 16 秒 |

视频按原文件逐字节保存，文件大小、时长与 SHA-256 见 [视频清单](docs/videos/manifest.json)。

此公开仓库仅保留当前展示所需的 Python 源码、必要测试、配置、固定模型和录屏；采用独立的初始提交，不包含原项目历史、论文全文、实测 bag 或训练日志。

## 当前模型与启动

当前展示使用已有训练组 `20260913-103135-634244600` 的 **U13773** 固定检查点：2 艘 ATK、3 艘 DEF、1 艘动态 TGT；ATK 使用 `attack` 参数组，DEF 与 TGT 共享 `defense` 参数组。U13773 是保存点的全局更新编号，不表示每一方都独立训练了同样多次。模型来源、配置与校验值见 [模型清单](docs/showcase-model.json)。本次整理没有重新训练或修改算法、场景与 UI。

Windows / PowerShell，安装 Git 和 uv 后，在准备存放项目的目录执行。固定模型随 Git 保存，无需另行下载：

```powershell
git clone https://github.com/lvdacong/AUV-MADRL-Showcase.git
Set-Location AUV-MADRL-Showcase\Python
uv sync --locked
Set-Location src
..\.venv\Scripts\python.exe -m auv_madrl goal-reach-viewer --port 8766 --checkpoint ..\..\Artifacts\GoalReach\20260913-103135-634244600-unified-match-seed42\checkpoint-013773.pt
```

在 Chrome 打开 [本地 Viewer](http://127.0.0.1:8766/)，确认 ATK、DEF、TGT 均显示已加载模型，再开启顶部 **Run** 查看。首次启动默认暂停；训练组旁的 **Train / Start** 用于训练，与预览的 Run 不同。Python 及依赖版本由 `Python/pyproject.toml` 与 `Python/uv.lock` 固定；当前展示在作者的 Windows 环境中验证，其他环境未验收。

在 `Python` 目录执行 `uv run pytest -q` 可运行随仓库提供的测试。发布前的环境、模型与文件校验记录见 [验证清单](docs/verification.json)。

## 署名与使用边界

Copyright © 2026 lvdacong. **保留所有权利 / All rights reserved.** 作者原创代码、文档、模型与录屏适用根目录 [LICENSE](LICENSE)；公开可读不额外授予改作、再分发或商业使用授权，复用请先联系作者，并保留来源与署名。Git 历史与文件校验值用于追溯版本，不是版权登记证书，也不保证杜绝剽窃。

Viewer 配乐按其原许可使用，见 [第三方素材说明](docs/THIRD_PARTY_NOTICES.md)。涉及视频中的第三方声音或画面时，作者声明只覆盖其原创部分。
