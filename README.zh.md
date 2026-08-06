# 活在后室

[English](README.md)

## 关于上游来源的说明

### 层级、物品、实体的入选

**一切内容的入选都必须符合本仓库的知识产权协议。**

选入原版游戏的层级、物品、实体主要来源是各站点的文章，包括但不局限于
[Wikidot](https://backrooms-wiki.wikidot.com/) 和
[Fandom](http://backrooms.fandom.com/wiki/)，亦不排除融合或原创。
入选游戏内容的不一定是该文章的最新版本，也不一定是该文章的原语言版本，
也不保证该版本未被有关站点删除。游戏亦可能对入选内容进行改编、修改、删减、融合。

### 层级、物品、实体的编号

**编号尽可能让玩家知道对应内容的来源。**

入选内容的编号一般采用其自身指定的编号。如果有两个不同内容使用同一个编号
且都入选游戏，游戏可能在编排上对其进行调整，方式包括但不限于添加后缀
（如 `level-xxx-wikidot` 和 `level-xxx-fandom`）。

对于来自特定语言站点的文章，一般采用该语言站使用的编号，如
[Wikidot中文站](https://backrooms-wiki-cn.wikidot.com/) 的“层级”使用
`level-c-xxx`。如有冲突，适用上段规则处理。

### 本地化

介于翻译之间和版本之间存在差异，本地化以被参考的原文所使用的语言为准，
而不使用对应语言站点的翻译作为唯一参考。

对于有多种翻译对应的语言，就一般情况而言可以任意选择更“好听”或更“符合含义”的版本。

但如果该语言的社群译名正确性有强烈要求，此类问题由该语言译者自行商议解决方案，
见对应语言的 README 文件。处理有关问题的原则是游戏内同一事物只能有一个名称。

### 中文翻译的版本问题

**提示：中文是本仓库的“母语”之一，这里所指的翻译主要是后室原生术语的翻译。**

中文存在 [Fandom翻译表](https://backrooms.fandom.com/zh/wiki/Backrooms_Wiki:译名参考列表)
和 [Wikidot翻译表](https://backrooms-wiki-cn.wikidot.com/different-translated-names)
两版翻译表。游戏根据来源文章和实际需要，确定了采用的翻译，没有与任何一版完全一致，
也没有保证两者均衡分配，情况见
[语言文件](game-data/vanilla/data/languages/zh-CN.toml)。如有异议，请提交 issue。

## 获取美术资产

本仓库不包含美术资产。具体获取方法见 [ASSETS.md](ASSETS.md)。

## 文档

项目说明文档放在 [docs/](docs/)：

- [TODO.md](docs/TODO.md) —— 未实现的功能与方向
- [optimizations.md](docs/optimizations.md) —— 确认过但暂缓的优化点
- [BUG.md](docs/BUG.md) —— 已知 bug、根因与当前方案

## 许可证

- 代码采用 GPL v3 许可，见 [LICENSE](LICENSE)。
- 剧情、层级创意使用 [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) 授权。
- 美术资产授权情况另行声明，见 [ASSETS.md](ASSETS.md)。

## AI 声明

本仓库的代码有 AI 辅助编写。
