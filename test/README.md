# 测试数据

本目录存放测试 / 演示用资产（模型、环境图等），**内容不入库**（见 .gitignore），
需要手动准备以下文件：

- `test.glb`：PBR 测试模型（组合扳手），CC0，来源：
  <https://polyhaven.com/a/combination_wrench>
- `test.hdr`：测试环境贴图（HDRI），用于环境管线（天空盒 + IBL）测试

文件缺失时相关测试会自动跳过（`is_file` 检查）；需要跑完整测试时请手动放入
上述两个文件。

---

# Test Data

This directory holds test / demo assets (models, environment maps, etc.).
Contents are **not committed** (see .gitignore); prepare these files manually:

- `test.glb`: PBR test model (combination wrench), CC0, from
  <https://polyhaven.com/a/combination_wrench>
- `test.hdr`: test HDRI environment map for the environment pipeline
  (skybox + IBL) tests

Tests that need a missing file skip themselves via an `is_file` check; put
both files here to run the full suite.
