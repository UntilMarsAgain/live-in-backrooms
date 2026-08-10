# TODO：未实现功能

记录还没实现的功能与方向，每项几句。已实现的优化/缺陷见同目录
`optimizations.md` / `BUG.md`。

## 当前优先级

1. **Level 0 拼图地形**：数据模型与生成（依赖物理刻的流式加载）。

## 渲染

- **阴影贴图**：PBR 目前没有阴影。从灯光视角渲染深度贴图（独立 pass），
  片元着色器采样比较即可；方向光优先。
- **后处理（Bloom 等）**：HDR 中间目标已就位（场景 pass → 色调映射 blit），
  可在两步之间插入 Bloom / SSAO / SSR 等后处理 pass；Bloom 也是自发光通道
  落地的前提（没有 Bloom，emissive 只能"发光"不能"发亮"）。
- **半透明物体**：管线 `blend: None`，全部不透明。加第二条管线
  （深度测试开、写入关、blend 开），透明物体从远到近排序。
- **UI 叠加层**：HUD、菜单、交互提示等 2D 叠加渲染，最终由 winit 事件驱动
  交互（与 FreeCameraController 的鼠标捕获协调）。
- **曝光统一**：色调映射已统一到 blit pass（AgX + 场景级 EV 窗口）；
  `intensity`（IBL 系数）与 `skybox_exposure`（天空盒亮度）已解耦；
  IBL 漫反射仍不经过曝光，与天空盒亮度未必一致——若需统一，把曝光乘法
  移到 blit（保持 AgX 输入为原始辐射值）。

## 游戏内容 / 关卡

- **Level 数据模型**：`App::load_scene` 一站式切换已实现（环境 + 资产 +
  实体），但关卡数据模型未定义——`Level { 场景, 环境, 环境强度, 出生点,
  灯光预设 }` 待落地，`load_level(level)` 把模组作者工作流落成代码。
- **程序化迷宫生成（Level 0）**：墙/地面网格生成、区块化加载与卸载；
  落地时一并做渲染批处理（见优化清单的 MultiDrawIndirect）。
- ~~**多 Level 切换**~~ 已实现基础版：`App::load_scene` 一站式切换——
  环境随场景热替换（`set_environment` / `reset_environment`）、旧场景实体
  级联 despawn、旧资产 `PinToken` 自动 unpin、渲染器不重建（F1/F2 demo
  切换即此路径）。关卡数据模型（`Level`）仍未定义，见"游戏内容 / 关卡"。

## 场景 / 物理

- ~~**物理刻调度器**~~ 已实现：`Playground` 内 `FixedTimestep` 累加器 +
  固定步长 `tick_schedule`（世界变换传播、自由相机），渲染刻独立按帧；
  资产 GC 由 App 按真实时间间隔触发（见"资产 / 资源"）。
- **静态场景模板的形态**：`Scene` 目前是 indextree 节点树 + 节点类型枚举
  （Empty/Mesh/Light/Camera），`Playground::load_scene` 逐节点翻译成 ECS
  实体（烘焙 Transform/MeshObject/LightC/Collider + ChildOf 层级）。这个
  "模板 → 实体"翻译层对加载期是够用的，但模板与 ECS 组件结构各自维护、
  语义重复（如碰撞盒从网格 AABB 派生这一步分散在 spawn 逻辑里）；关卡
  切换与程序化生成都要经过它。将来可考虑更贴近 ECS 的模板形态（如组件
  bundle 描述 / 直接声明式产出实体），让翻译层消失或变薄。
- **物理动力学**：AABB 粗碰撞已落地（点包含 / 物体碰撞 / 外部探针 / 相机
  分轴滑动 / 碰撞箱调试显示）；未做刚体动力学（角色移动、投掷物）与
  更贴合的碰撞体（凸包 / 三角网格 + BVH，地形建模网格需要）。
- **可交互物**：相机以外的场景对象（开关、拾取物、触发器）与交互 API。

## 资产 / 资源

- ~~**路径 → 句柄**~~ 已完成：`GamePath` 规范化 + 反向索引（路径 → 句柄）、
  同路径同类型去重、remove 同步删除。
- ~~**合并资源空间多包合并**~~ 已实现：包发现（扫描 `game-data/` 下含
  `package.toml` 的目录）+ 依赖/冲突校验 + 顺序 reconcile（`PackConfig`）
  → `MergedResourceSpace::from_pack_roots` 按优先级覆盖解析。待补：zip
  压缩包支持、namespace 目录级叠加（当前是文件级覆盖）。
- **智能 GC**：`AssetManager::gc` / `GpuManager::gc` 都是**纯成员操作**，
  统一实现在 [`GcPolicy::should_keep`]（`core/gc.rs`），两侧只喂自己的时钟
  与 [`GcInfo`] 记录；扩展只改 `gc.rs`。App 按真实时间间隔（`GC_INTERVAL`）
  自动调用；内存占用检测 + 预算阈值 + 超限强制全量清扫已实现
  （sysinfo 跨平台读取物理内存，环境变量可覆盖）。
- **pin 引用计数**：已从开关改为引用计数 + `PinGuard` RAII 守卫（CPU 侧）；
  批量驻留用 `PinToken`（`Weak<Mutex<AssetManager>>`，场景实例持有、
  覆盖即自动 unpin）；GPU 侧驻留按最近使用窗口淘汰（gc 不依赖 CPU pin，
  纯成员），淘汰后自愈重传。
- ~~**关卡级资产清单**~~ 已实现基础版：`PinToken` 批量 pin 场景引用的
  网格/贴图（去重），由 `SceneInstance` 持有、切换即自动 unpin。
  `Level` 数据模型与按关卡显式清单管理待 `Level` 落地后再细化。

## 音频

- **音频系统**：后室氛围音、脚步、环境声；资源与场景绑定，随 Level 加载。

## 性能 / 工程

- **脏标记 + 按需重绘**：现在每帧 `request_redraw` 持续渲染；场景无变化时
  停止重绘，静止时 CPU 占用归零。
- **MultiDrawIndirect**：区块世界几千个可见区块时，从 CPU 逐条 draw 改为
  GPU 批量（能力探测 + 回退）。
- ~~**异步加载**~~ 已实现（**文件级、多类型**）：`AssetManager::load_file_async`
  按 `FileLoader::scan` 先注册各类型占位句柄（Loading），后台线程 `parse`
  **完整解析一次**后填充；`get` 阻塞等待；`handle_state`/`status` 可查询。
  glb 用 `GlbFileLoader`（同步 `load_file`/`load_scene` 仍可用）；上传仍在主线程。
- **纹理 mipmap**：当前 1 级 mip，远处纹理会闪烁。
