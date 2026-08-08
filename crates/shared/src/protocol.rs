//! C/S 同步协议：服务端权威状态 → 客户端本地解析。
//!
//! 原则：跨进程传输的是**实体 ID / 模型来源（GamePath）与变换**，绝不传顶点
//! 数据。客户端在本地合并资源空间里按 GamePath 解析资产，因此两边各自持有
//! 的资源空间副本必须一致（握手校验 ID 与顺序就是防这个）。
//!
//! 单机集成模式下，快照走内存队列（`mpsc`），传的是对象本身、无序列化；
//! 真联机时把队列换成 socket，在边界序列化本模块的类型即可，其余代码不动。

use crate::core::game_path::GamePath;
use crate::core::light::Light;
use crate::core::transform::Transform;

/// 实体类型：服务端只描述"是什么 + 摆在哪"，具体渲染数据客户端本地解析。
#[derive(Debug, Clone, PartialEq)]
pub enum EntityKind {
    /// 网格实体：模型来源（GamePath），客户端在本地资源空间解析并上传 GPU。
    Mesh(GamePath),
    /// 灯光实体：类型/颜色/强度 + 位置朝向（变换）。
    Light(Light),
}

/// 单个实体的权威状态（服务端每 tick 产出）。
#[derive(Debug, Clone, PartialEq)]
pub struct EntityState {
    /// 服务端分配的稳定实体 ID（快照内唯一；删除后不立即复用）。
    pub id: u32,
    pub kind: EntityKind,
    pub transform: Transform,
}

/// 一帧权威世界快照。
#[derive(Debug, Clone)]
pub struct WorldSnapshot {
    /// 服务端 tick 序号（从 1 递增）。
    pub tick: u64,
    pub entities: Vec<EntityState>,
}
