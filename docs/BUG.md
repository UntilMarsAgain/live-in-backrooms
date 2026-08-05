# BUG 记录

当前依赖或实现中存在的 bug：现象、根因、现在的方案。
性能/优化债见 [optimizations.md](./optimizations.md)。

## wgpu GL 后端：数组纹理 readback 拷贝全零

**这是一个wgpu实现的bug，需等待上游修复（见[#10015](https://github.com/gfx-rs/wgpu/issues/10015)）**

- 现象：GL 后端下环境贴图（天空盒/IBL）全黑，HDR 数据与 CPU 数学正常；
  同样的代码在 Vulkan/Metal 上正常（`examples/vulkan_probe.rs` 实测确认）。
- 根因：wgpu 的 GL 后端（llvmpipe 实测）对 2D 数组纹理的
  `copy_texture_to_buffer` 读回不可靠——整块拷贝与逐层拷贝都返回全零；
  `examples/gl_probe.rs` 实测存储写入、数组采样、逐层上传均正常
  （测试 C/E/F OK），唯独数组读回（A/A2/B/D）全零。
- 方案：按后端分流——Vulkan/Metal 走 GPU 计算，GL 等回退 CPU 转换 +
  逐层 `write_texture` 上传（`EnvConversionPath`，启动日志可见）。
  回退路径全程不读回数组纹理，绕开该 bug。

## GPU 转换只写环境图左上角（参数缓冲分时复用）

- 现象：Vulkan 上 GPU 路径天空盒仍黑；单 pass 探针却正常。
- 根因：`queue.write_buffer` 立即入队、先于 `submit()` 里的 pass 执行；
  两个计算 pass 复用同一参数缓冲，第二个写入先覆盖，pass1 读到错误参数
  （size=32），256² 环境图只有左上角 32×32 有数据。
- 方案：两个计算 pass 各用独立参数缓冲（`env_convert_params` /
  `irradiance_params`）。
