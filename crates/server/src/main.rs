//! 服务端 headless 入口：不依赖任何渲染/窗口 crate（依赖图里无 wgpu/winit）。
//!
//! 这里演示同一份 [`Server`] 代码在无窗口环境独立运行：固定 tick 推进
//! 权威世界并打印快照。单机集成时客户端直接嵌入这份代码（`Server::spawn_thread`），
//! 真联机时把快照改成网络发送即可。

use std::time::Duration;

use lbr_server::Server;

fn main() {
    let mut server = Server::new();
    println!("服务端启动：硬编码测试世界（实体定义 TOML 落地前）");
    for i in 0..20 {
        let snap = server.tick(1.0 / 20.0);
        if i % 5 == 0 {
            let mesh_count = snap
                .entities
                .iter()
                .filter(|e| matches!(e.kind, lbr_shared::EntityKind::Mesh(_)))
                .count();
            println!(
                "tick {:>3}：{} 个实体（{} 个网格）",
                snap.tick,
                snap.entities.len(),
                mesh_count
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
