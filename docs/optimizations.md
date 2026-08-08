# 优化清单

这份文档记录项目开发中"确认过、但现阶段不做"的优化点：为什么现在不做、将来怎么做、
以及哪些路已经铺好（做的时候不用推倒重来）。原则是**静态数据加载时解决，动态数据
渲染时解决**——先不为了用不上的优化增加复杂度，但给它们留好位置。
维护规则：**已实现的优化移出本文档**（不留已完成条目）；**明确不做的用删除线标记**
留在原处。

## 物体世界矩阵/法线矩阵：每帧全量上传

- **现状**：每帧对每个节点算 `world_transform`（O(N×深度)）并整体 `write_buffer` 一次。
- **问题**：大部分物体不动，重复推导是浪费；但物体是"碰巧没变"而非"保证不变"
  （游戏实体将来会动），不能像资产那样硬性只传一次。
- **已铺好的路**：`write_buffer` 支持偏移，可只更新脏物体；场景图已有代际句柄和
  `reparent`，加"transform 版本号 + 子树脏标记"只需给 `SceneObject` 加个字段。
- **触发时机**：物体数量上千、profile 显示世界矩阵计算成为热点时。先做 CPU 侧脏传播
  缓存，再考虑 GPU 侧跳过整次写入；更远期可把静态/动态物体拆两组 + instancing。

## 灯光：三种类型已落地，面光是近似

- **现状**：方向光 / 点光 / 面光（矩形面板）三种类型，统一 80 字节/灯；
  点光平方反比衰减，面光按"朗伯发射面板 + 平方反比"近似；
  uniform 数组定长 8（656 字节），CPU 收集时 clamp。
- **待补**：真实矩形面光需要 **LTC**（BRDF LUT + 多边形积分）；聚光灯；
  强度/颜色目前是直觉单位，没有 lux/W 物理语义。
- **已铺好的路**：`collect_lights` 独立成函数、`size` 字段已为 LTC 预留；
  灯数超过几十时换 storage buffer（`var<storage>` + 运行时长度数组）即可去掉上限。

## 网格资产：新增时整体重传

- **现状**：`MeshLibrary` 只追加、带版本号；`upload_meshes` 版本变了就把全部网格
  合并缓冲整体重建（旧缓冲 drop，wgpu 延迟销毁）。
- **问题**：运行时新增资产会有一瞬重传（调色盘量级只是几毫秒级，无感）。
- **已铺好的路**：`MeshLibrary.version` 机制可扩展到灯光/物体；`write_buffer` 支持
  偏移，增量上传只传新增网格的区间。
- **触发时机**：运行中高频注册资产、或上传量级大到出现可见卡顿。多 Level 后若显存
  吃紧，把"永久驻留"改成"按 Level 分组的驻留/淘汰"（bundle 模式），形状不变。

## 网格句柄依赖"只追加不删除"

- **现状**：`MeshKey` 是稠密编号，`mesh_ranges[handle.index()]` 一跳直达，靠"库只追加
  不删除"保证编号稳定。
- **问题**：一旦需要删除资产（比如做资产淘汰），稠密编号会失效。
- **已铺好的路**：届时换成代际句柄 + `handle_to_range` 映射表即可， 渲染侧改动集中在 
  `upload_meshes`。
- **触发时机**：引入资产淘汰/卸载时。

## 渲染批次：每物体一次 draw

- **现状**：每个网格物体一次 `set_bind_group` + `draw_indexed`（几百次调用以内没问题）。
- **优化方向**：静态几何合并成大缓冲按区间画（区块化时自然到来）、重复物体 instancing
  （管线加实例输入即可，`draw_indexed(..., 0..instances)` 原生支持）。
- **触发时机**：Level 0 迷宫/区块体系落地时一起做。

## MultiDrawIndirect：从 CPU 逐条画改为 GPU 批量画

- **现状**：所有绘制仍是 CPU 驱动（每物体一条命令），区块世界里几千个可见区块 =
  几千次 CPU 命令编码往返。
- **优化方向**：参考 Minecraft 26.3 Snapshot 6 的做法——地形区块改用 MultiDrawIndirect，
  每帧 CPU 只上传一张紧凑的间接参数表（每个 draw 5×u32：
  `index_count / instance_count / first_index / base_vertex / first_instance`），
  然后一条 `multi_draw_indexed_indirect` 让 GPU 自己遍历完成全部绘制。
  per-draw 数据（模型矩阵、法线矩阵、材质索引等）从"动态 offset uniform"改为
  storage buffer，用 `@builtin(instance_index)`（配合 `first_instance` 当 draw ID 的技巧）
  取自己那份；按材质分组，每组一条 multi-draw。实体等可动资产同样适用：不同网格
  由 `first_index`/`base_vertex` 指向大缓冲中的不同区间，蒙皮实体额外需要一块
  骨骼矩阵缓冲 + per-draw 关节区间偏移。
- **上传模型（"批量上传"）**：几何顶点只在块加载/变更时进 GPU；每帧的动态数据是
  **一次 `write_buffer` 写入的紧凑表**（间接参数 + storage 里的 per-draw 变换/材质），
  一次覆盖成百上千个 draw，而不是逐块上传几何——这就是 26.3 Snapshot 6 terrain
  重构后的形态（terrain shader 支持 multi-draw；`DynamicTransforms` 重排 uniform
  内存、减少 std140 padding；另有 OIT 相关的 `RENDERPEARL_EXPLICIT_DEPTH_INVARIANCE`）。
- **约束**：需要 `Features::MULTI_DRAW_INDIRECT`（用 `first_instance` 还需要
  `INDIRECT_FIRST_INSTANCE`），不是所有后端都支持 → 保留现有 CPU 循环做回退
  （能力探测后二选一）。
- **已铺好的路**：调色盘按材质分组天然对应"每组一条 multi-draw"；
  `MeshLibrary` 的大顶点/索引缓冲 + `mesh_ranges` 区间正好映射到
  `first_index`/`base_vertex`；`Vertex::layout()` 扩展性已就位。
- **参考**：Minecraft 26.3 Snapshot 6 公告（含 terrain shader 为 multi-draw 重构、
  `DynamicTransforms` 内存重排、OIT 相关的 `RENDERPEARL_EXPLICIT_DEPTH_INVARIANCE`）：
  https://www.minecraft.net/en-us/article/minecraft-26-3-snapshot-6
- **触发时机**：Level 0 迷宫/区块体系落地时，与上一条一起做；设计成引擎通用批处理层
  （地形 + 实体 + 将来粒子共用），而不是地形专用。

## 着色器：PBR + IBL 已落地，场景反射/阴影待补

- **现状**：GGX 分布 + Smith 几何 + Schlick 菲涅尔（metallic workflow），
  法线贴图（切线空间 → 世界，Gram-Schmidt）、金属度/粗糙度贴图已接入。
- **环境管线（已落地）**：HDRI → 环境立方体贴图（256²×6）→ 辐照度图（32²×6）
  → 镜面预过滤图（128²×6，8 级 mip，GGX 重要性采样）+ BRDF LUT（128²）
  → 天空盒 + mesh @group(4) 漫反射/镜面 IBL；环境转换全部走 GPU 计算
  （仅 PRIMARY 后端，无 GL 回退），`FLOAT32_FILTERABLE` 不支持时回退点采样。
- **已知简化（Phase 1 的偷懒）**：
  - 镜面反射只采样环境图（天空盒 HDR），**场景物体不参与反射**，且反射无遮挡
    （会"穿透"墙壁看到环境），见下方"镜面反射"条目；
  - 预过滤图按 mip 分层粗糙度，但环境立方体贴图本身仍是单级（无 mip）；
  - IBL 漫反射不经过曝光，与天空盒亮度未必一致（曝光拆分见
    "HDR 中间目标 + 色调映射 blit"条目）；
  - 环境转换在启动时阻塞完成（加载画面出现前），未异步化；
  - 环境是"每 Level 一份"的资产，当前只有启动时加载一次，没有 Level 切换
    的卸载/热替换（`set_environment` 已支持重建，App 层尚未接关卡数据）。
- **待补**：阴影贴图（从灯光视角渲染深度，独立 pass）；场景反射（反射探针 /
  SSR，见下方条目）。

## HDR 中间目标 + 色调映射 blit（已落地）

- **现状**：场景 pass（网格 + 天空盒 + 调试线框）渲染到 Rgba16Float 离屏
  纹理（原始辐射值，可 >1），色调映射 blit pass 采样它做 AgX 映射后写
  交换链（`blit.wgsl` / `blit.rs`）；全帧只做一次色调映射，天空盒与物体
  走同一条曲线。窗口 resize 时 HDR 纹理与 blit 绑定组随深度缓冲一起重建。
- **已铺好的路**：后处理（Bloom / SSAO / SSR）可以在"场景 pass 之后、
  blit 之前"插入，直接消费 HDR 目标；阴影等需要额外视角的 pass 也可复用
  同一套离屏纹理框架。
- **已知简化**：`environment_params.intensity` 仍兼任 IBL 系数与天空盒曝光，
  blit 不乘曝光（见 `blit.wgsl` 注释）——曝光拆分时应把乘法统一移到 blit；
  调试线框现在也走色调映射（画进 HDR 目标，与场景一起被映射）。

## 镜面反射：只反射环境图，不反射场景物体

- **现状**：镜面 IBL（预过滤 mip 链 + BRDF LUT）已落地，金属物体能反射
  "天空"方向的环境光；但反射只查预过滤环境图，**场景里的墙、家具、其他实体
  不会出现在反射中**，且反射方向和漫反射 IBL 一样没有遮挡（反射会穿透墙壁）。
- **方向（按推荐顺序）**：
  1. **反射探针（Reflection Probe）**：在房间/区域中心把场景渲染成立方体贴图，
     复用现有的预过滤/BRDF 采样链路；以房间为单位的封闭空间（后室）最合适，
     而且探针天然只捕获房间内部，**同时解决"看不到物体"和"穿透墙壁"两个问题**。
     探针就是关卡数据里的一个对象（位置 + 范围 + 更新策略），程序化生成时
     自动放置，契合数据驱动设计。
  2. **屏幕空间反射（SSR）**：在主渲染后的深度/法线上做光线步进，反射屏幕内
     可见的几何；需要深度/法线缓冲 + 一条后处理 pass，屏幕外物体反射不到，
     粗糙表面会噪点、需要模糊。作为探针的屏幕级补充。
  3. **光线追踪**：真正逐光线的场景反射，质量最高，需要 RT 硬件或实验性 API，
     明显靠后。
- **已铺好的路**：`EnvironmentGpu` 已有预过滤纹理 + BRDF LUT + mesh 绑定组，
  探针只需把"环境立方体贴图"换成"探针捕捉的场景 cubemap"，管线与着色器
  完全复用；`set_environment` 的重建路径可作为探针热更新的雏形。
- **触发时机**：金属物体 / 镜面表面成为视觉重点，或室内反射违和感明显时；
  探针建议和 Level 0 区块体系一起落地（生成时放置探针）。

## 纹理：基础色 / 金属度粗糙度 / 法线已接入

- **现状**：`TextureLibrary`（只追加、版本号驱动增量上传）+ `Material`
  （base_color / metallic / roughness / normal 因子与贴图）已落地；glTF 的
  `baseColorTexture` / `metallicRoughnessTexture`（B=金属度、G=粗糙度）/
  `normalTexture` 加载并采样，缺贴图时用 1×1 兜底纹理（白 / 中性法线）；
  顶点新增 TANGENT（法线贴图 TBN）。
- **待补通道**：自发光（emissive）、AO（occlusion）、顶点色之外的材质混合——
  同样的"多一个采样器 + 多一个绑定"模式。
- **切线**：文件自带 TANGENT 直接用；缺失时按 MikkTSpace（`mikktspace` crate，
  Blender 同款算法）自动计算，UV 接缝处正确。因此**不需要**在 Blender 手动
  导出切线；资产也无需先三角化——glTF 数据本身就是三角形（Blender 导出时已
  隐式三角化），我们的 MikkTSpace 跑在三角形上不会报错。Blender 的切线导出
  报 "Could not calculate tangents" 是因为它在三角化之前对含 ngon 的原始网格
  算切线；只有想让 .glb 文件自带 TANGENT（供其他工具用）时才需要先手动三角化。
- **待办**：mipmap 生成（现在是 1 级，远处纹理会闪烁）、纹理驻留/卸载
  （多 Level 后）、UV 的 v 翻转约定验证（glTF 与 wgpu 采样原点实测为准）。

## 半透明物体

- **现状**：管线 `blend: None`，全部不透明。
- **已铺好的路**：深度缓冲已就位；届时加第二条管线（深度测试开、写入关、blend 开），
  透明物体从远到近排序，复用同一张深度附件。
- ~~**备选**：OIT（顺序无关透明）~~复杂度高，明确不做。

## 深度精度

- **现状**：near=0.1 / far=100，`Depth24Plus`。
- **风险**：场景范围变大后远处 z-fighting（深度是 1/z 双曲线，远处精度差）。
- **已铺好的路**：管线的 `DepthBiasState` 已配；调近远平面比例或加深度偏差即可。
- **触发时机**：Level 0 大场景出现闪烁时。

## 异步加载、多线程与加载提示

- **现状**：glTF 加载/网格上传都在主线程、启动时一次性完成。
- **已铺好的路**：资产不可变 + 全局库，将来异步加载时用 `Arc<Mesh>` 传递所有权
  （所有权分裂时 Arc 才值得用）；上传仍在主线程，构建/解析走工作线程。
- **触发时机**：加载时间开始影响体验时。

## 顶点格式扩展

- **现状**：POSITION / NORMAL / TEXCOORD_0 / COLOR_0（56 字节/顶点）。
- **已铺好的路**：`Vertex::layout()` 显式 offset，加 TANGENT、JOINTS_0/WEIGHTS_0
  （蒙皮）、第二套 UV 时只动 `Vertex` + 布局函数 + 加载器。
- **索引**：统一 u32；海量小网格时再考虑 u16。

## ~~负缩放/镜像节点约定~~

- **现状**：管线 CCW 背面剔除；glTF 若带负缩放（镜像）会翻转绕序导致面消失。
- **约定**：外部资产文件不提供负缩放（已确认）。若未来必须支持，需对镜像节点
  关闭剔除或修正绕序。

**提示：在blender中应用变换，可以直接调整网格**

## 每帧 request_redraw（持续渲染）

- **现状**：`ControlFlow::Wait` + `update()` 每帧 `request_redraw()`，即使场景静止
  也在连续渲染。
- **优化**：配"场景无变化"版本号后，可改成"有变化才重绘"，静止时 CPU 占用归零。
- **触发时机**：想省电/省 CPU 时，和脏标记机制一起做。

## 测试覆盖

- **现状**（26 个测试）：
  - CPU 单测：场景世界矩阵、glTF 加载、环境转换（`to_cubemap` / `irradiance_map`），
    不碰 GPU，任何环境可跑；
  - **WGSL 语法校验**：用 naga 解析并校验 `mesh.wgsl` / `environment.wgsl`，
    语法与绑定组声明错误在 `cargo test` 阶段暴露（`cargo build` 不编译 WGSL）；
  - **无头冒烟测试**：不创建窗口，请求软件渲染设备（llvmpipe GL），真跑
    环境资源创建 → 转换 → 天空盒渲染到离屏 → 读回像素验证非黑；无 GPU
    环境自动跳过并打印原因；
  - **端到端像素验证**：真实 `test/test.hdr` 转换后渲染天空盒，断言平均亮度非黑。
  - **镜面 IBL 读回验证**：GPU 路径下读回预过滤 mip 0 与 BRDF LUT，断言非黑
    （防"参数缓冲复用导致预过滤图全黑"这类回归）。
- **注意事项**：llvmpipe 软件渲染器并行跑多个 GPU 测试会段错误（线程问题），
  因此 GPU 相关测试统一用 `cargo test -- --test-threads=1` 跑；单线程下全部通过。
- **可补**：`collect_lights`（方向推导）、`upload_meshes` 合并区间、uniform 布局
  （大小/偏移断言）、strip/fan 转换的边界。
- **后端能力验证**：`examples/vulkan_probe.rs`（强制 Vulkan，A/B/C/D 四项实测），
  换机器/驱动后跑一遍即可确认后端能力，避免静默依赖 storage 数组纹理等
  不可靠特性。
