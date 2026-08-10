//! ECS：场景 / 渲染 / 输入统一的实体组件系统（实验分支）。
//!
//! 存储、查询、系统与调度执行由 bevy_ecs 提供；**执行时机由本模块的
//! [`FixedTimestep`] 累加器控制**——物理刻按固定步长跑 [`tick_schedule`]，
//! 渲染刻按帧跑 [`frame::render_schedule`]（本模块定义）。
//!
//! 现有 [`Scene`](crate::engine::scene::Scene) 保留为**加载期模板**
//! （glTF / 演示构建、合并、环境），由 [`playground::Playground::load_scene`]
//! 生成进 [`World`]；渲染与自由视角相机都改为消费 ECS 组件 / 系统。

pub mod camera;
pub mod components;
pub mod frame;
pub mod hierarchy;
pub mod playground;

use std::collections::HashSet;
use std::time::Duration;

use bevy_ecs::prelude::*;
use winit::keyboard::KeyCode;

/// 固定步长累加器：真实帧时间累积到步长就补一个物理刻。
#[derive(Debug, Clone, Copy)]
pub struct FixedTimestep {
    step: Duration,
    acc: Duration,
}

impl FixedTimestep {
    pub fn new(step: Duration) -> Self {
        Self {
            step,
            acc: Duration::ZERO,
        }
    }

    /// 推进一帧真实时间，返回（应补的物理刻数, 渲染插值 alpha ∈ [0,1)）。
    pub fn advance(&mut self, frame_dt: Duration) -> (u32, f32) {
        self.acc += frame_dt;
        let step_secs = self.step.as_secs_f64();
        let ticks = (self.acc.as_secs_f64() / step_secs) as u32;
        self.acc -= Duration::from_secs_f64(step_secs * ticks as f64);
        let alpha = (self.acc.as_secs_f64() / step_secs) as f32;
        (ticks, alpha)
    }
}

/// 固定步长（物理刻 dt），作为资源供系统读取。
#[derive(Resource, Debug, Clone, Copy)]
pub struct FixedStep(pub Duration);

/// 每物理刻的输入快照：事件回调只写这里（作为资源放进 [`World`]），系统只读。
///
/// 取代旧的 `InputController` 回调式控制器：App 层事件处理只累积状态，
/// 相机系统在固定刻消费增量（`look_delta` / `scroll_delta` 消费后清零）。
#[derive(Resource, Debug, Default)]
pub struct InputSnapshot {
    /// 当前按下的键。
    pub keys: HashSet<KeyCode>,
    /// 待应用的鼠标旋转量（像素；系统消费后清零）。
    pub look_delta: (f32, f32),
    /// 待应用的滚轮位移（格数；系统消费后清零）。
    pub scroll_delta: f32,
}

impl InputSnapshot {
    pub fn pressed(&self, code: KeyCode) -> bool {
        self.keys.contains(&code)
    }
}

/// 调试可视化开关（App 按键切换，渲染刻系统读取后写入渲染指令）。
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct DebugFlags {
    pub show_light_debug: bool,
    pub show_collision_debug: bool,
}

/// 物理刻调度：世界变换传播 → 自由相机（固定步长跑这个）。
pub fn tick_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems((hierarchy::propagate_world_transforms, camera::free_camera).chain());
    schedule
}
