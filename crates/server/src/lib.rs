//! 服务端权威模拟（可嵌入客户端进程，也可独立 headless 运行）。
//!
//! 职责：
//! - 固定 tick 推进权威世界，每 tick 产出 [`WorldSnapshot`]；
//! - 单机集成：`Server::spawn_thread` 把模拟放到独立线程，经 `mpsc` 快照
//!   队列交给渲染线程（无序列化）；真联机时同一份代码可改为网络发送；
//! - 实体定义（TOML）尚未落地，当前实体是硬编码的测试内容。

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use glam::{Quat, Vec3};
use lbr_shared::core::light::Light;
use lbr_shared::core::transform::Transform;
use lbr_shared::protocol::{EntityKind, EntityState, WorldSnapshot};
use lbr_shared::GamePath;

/// 权威世界（服务端）。
///
/// 只持有 sim 需要的东西（实体状态），不持有任何渲染/资产状态；
/// 资产解析由消费方（客户端）按实体里的 GamePath 在本地完成。
pub struct Server {
    tick: u64,
    entities: Vec<EntityState>,
}

impl Server {
    /// 新建权威世界：硬编码测试实体（实体定义 TOML 落地后替换）。
    ///
    /// 内容与客户端早期 demo 的布光一致：方向光 + 点光 + 面光，
    /// 加上一个绕 Y 轴缓慢旋转的测试模型（`test:test.glb`）。
    pub fn new() -> Self {
        // 与 demo 相同的"光从右上前方照来"约定（局部 -Z = 光行进方向）。
        let light_arrival = Vec3::new(0.5, 0.6, 0.6).normalize();
        let mesh: GamePath = "test:test.glb".parse().expect("内置测试路径合法");
        Self {
            tick: 0,
            entities: vec![
                EntityState {
                    id: 1,
                    kind: EntityKind::Light(Light::directional(Vec3::ONE, 0.7)),
                    transform: Transform::new(
                        Vec3::ZERO,
                        Quat::from_rotation_arc(Vec3::NEG_Z, -light_arrival),
                        Vec3::ONE,
                    ),
                },
                EntityState {
                    id: 2,
                    kind: EntityKind::Light(Light::point(Vec3::new(1.0, 0.85, 0.6), 18.0)),
                    transform: Transform::new(
                        Vec3::new(2.2, 1.8, 0.8),
                        Quat::IDENTITY,
                        Vec3::ONE,
                    ),
                },
                EntityState {
                    id: 3,
                    kind: EntityKind::Light(Light::area(
                        1.5,
                        0.6,
                        Vec3::new(0.9, 0.95, 1.0),
                        20.0,
                    )),
                    transform: Transform::new(
                        Vec3::new(1.8, 2.8, -0.8),
                        Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                        Vec3::ONE,
                    ),
                },
                // 测试模型：放在演示物体右前方，放大 5 倍，服务端每 tick 旋转。
                EntityState {
                    id: 4,
                    kind: EntityKind::Mesh(mesh),
                    transform: Transform::new(
                        Vec3::new(1.8, 0.0, -1.2),
                        Quat::IDENTITY,
                        Vec3::splat(5.0),
                    ),
                },
            ],
        }
    }

    /// 推进一个固定 tick（`dt` 秒），返回该 tick 结束后的世界快照。
    pub fn tick(&mut self, dt: f32) -> WorldSnapshot {
        self.tick += 1;
        // 演示：网格实体绕 Y 轴旋转——证明场景是服务端线程驱动的。
        for entity in &mut self.entities {
            if matches!(entity.kind, EntityKind::Mesh(_)) {
                entity.transform.rotation *=
                    Quat::from_rotation_y(dt * 0.6);
            }
        }
        WorldSnapshot {
            tick: self.tick,
            entities: self.entities.clone(),
        }
    }

    /// 以固定频率在独立线程运行，返回快照接收端（单机集成模式）。
    ///
    /// - 有界队列（容量 1）+ `try_send`：渲染线程跟不上时丢弃旧快照，
    ///   "最新状态优先"，不会积压；
    /// - 线程随进程退出而终止（骨架阶段无优雅停机；需要时加停止标志）。
    pub fn spawn_thread(hz: f64) -> mpsc::Receiver<WorldSnapshot> {
        let (tx, rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let mut server = Self::new();
            let period = Duration::from_secs_f64(1.0 / hz);
            let step = 1.0 / hz as f32;
            loop {
                let start = Instant::now();
                let snapshot = server.tick(step);
                // 满了就丢：渲染线程读的是"最新一帧"，不是逐帧序列。
                let _ = tx.try_send(snapshot);
                let elapsed = start.elapsed();
                if elapsed < period {
                    thread::sleep(period - elapsed);
                }
            }
        });
        rx
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_increments_and_rotates_mesh() {
        let mut server = Server::new();
        let first = server.tick(1.0 / 30.0);
        assert_eq!(first.tick, 1);
        assert_eq!(first.entities.len(), 4);

        let mesh = first
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::Mesh(_)))
            .expect("应有一个网格实体");
        let yaw0 = mesh.transform.rotation;

        let second = server.tick(1.0 / 30.0);
        assert_eq!(second.tick, 2);
        let mesh = second
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::Mesh(_)))
            .expect("应有一个网格实体");
        assert_ne!(mesh.transform.rotation, yaw0, "网格应随 tick 旋转");
    }

    #[test]
    fn spawn_thread_yields_snapshots() {
        let rx = Server::spawn_thread(60.0);
        let snap = rx.recv_timeout(Duration::from_millis(500)).expect("应有快照");
        assert!(snap.tick >= 1);
        assert_eq!(snap.entities.len(), 4);
    }
}
