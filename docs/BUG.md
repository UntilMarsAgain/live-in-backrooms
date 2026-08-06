# BUG 记录

当前依赖或实现中存在的 bug：现象、根因、现在的方案。
性能/优化债见 [optimizations.md](./optimizations.md)。

## GPU 转换只写环境图左上角（参数缓冲分时复用）

- 现象：Vulkan 上 GPU 路径天空盒仍黑；单 pass 探针却正常。
- 根因：`queue.write_buffer` 立即入队、先于 `submit()` 里的 pass 执行；
  两个计算 pass 复用同一参数缓冲，第二个写入先覆盖，pass1 读到错误参数
  （size=32），256² 环境图只有左上角 32×32 有数据。
- 方案：两个计算 pass 各用独立参数缓冲（`env_convert_params` /
  `irradiance_params`）。

## 已修复

### wgpu GL 后端数组纹理读回全零（已通过移除 GL 后端规避）

- 现象/根因：wgpu 的 GL 后端（llvmpipe 实测）对 2D 数组纹理的
  `copy_texture_to_buffer` 读回全零，导致环境贴图（天空盒/IBL）全黑；
  上游 issue [#10015](https://github.com/gfx-rs/wgpu/issues/10015)。
- 方案：游戏改用 `Backends::PRIMARY`（Vulkan / Metal / DX12），不再启用
  OpenGL；原 CPU 回退转换路径已删除，环境转换全部走 GPU 计算。

### 切换到无环境场景后残留上一关卡的天空盒

- 现象：先加载带环境（天空盒 + IBL）的场景，再切换到 `environment = None` 的
  场景，画面仍显示上一关卡的天空盒，不会回到默认黑环境。
- 根因：`App::load_scene` 只在场景**带**环境时调用 `Renderer::set_environment`，
  无环境时不重置；渲染器侧环境图始终存在（未设置时为 1×1 黑环境）。
- 方案：新增 `Renderer::reset_environment`，加载不带环境的场景时切回默认黑环境，
  并恢复默认环境强度与 AgX 窗口。
