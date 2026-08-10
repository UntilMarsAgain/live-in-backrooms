//! 实验：bevy_ecs 存储/查询 + 手动固定步长调度（方式一）。
//!
//! 只验证"`World`/`Query`/`System` 由 bevy 提供、**执行时机由自己控制**"的
//! 最小闭环：
//! - [`FixedTimestep`]：固定步长累加器（物理刻），返回补刻数与渲染插值 alpha；
//! - `tick_schedule`：物理系统（读输入快照 → 积分移动）；
//! - `render_schedule`：按帧消费（收集世界矩阵，模拟渲染端构建 ObjectData）。
//!
//! 独立实验：不接 App、不动现有 Scene/渲染器。确认可行后再决定迁移范围。

use std::time::Duration;

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3};

use super::core::data::transform::Transform;

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

    pub fn step(&self) -> Duration {
        self.step
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

/// 运动物体组件：复用现有 [`Transform`]，再加一个移动速度。
#[derive(Component, Debug, Clone)]
pub struct Motion {
    pub transform: Transform,
    pub speed: f32,
}

impl Motion {
    pub fn at(transform: Transform, speed: f32) -> Self {
        Self { transform, speed }
    }
}

/// 每物理刻的输入快照：事件回调只写这里，系统只读。
#[derive(Resource, Debug, Default)]
pub struct InputSnapshot {
    /// 期望移动方向（世界系；demo 不归一化，直接按速度缩放）。
    pub move_dir: Vec3,
}

/// 渲染端消费结果：按帧收集的物体世界矩阵（模拟 ObjectData 构建）。
#[derive(Resource, Debug, Default)]
pub struct RenderedMatrices(pub Vec<Mat4>);

/// 物理系统：按输入方向积分移动。
fn apply_input_movement(
    step: Res<FixedStep>,
    input: Res<InputSnapshot>,
    mut query: Query<&mut Motion>,
) {
    let dt = step.0.as_secs_f32();
    for mut motion in &mut query {
        let speed = motion.speed;
        motion.transform.position += input.move_dir * speed * dt;
    }
}

/// 渲染消费系统：把运动物体的世界矩阵收集进资源。
fn collect_world_matrices(query: Query<&Motion>, mut out: ResMut<RenderedMatrices>) {
    out.0 = query.iter().map(|m| m.transform.to_mat4()).collect();
}

/// 装配一个最小世界：固定步长 + 输入 + 渲染输出 + 一个演示实体。
pub fn demo_world() -> World {
    let mut world = World::new();
    world.insert_resource(FixedStep(Duration::from_secs_f64(1.0 / 60.0)));
    world.insert_resource(InputSnapshot::default());
    world.insert_resource(RenderedMatrices::default());
    world.spawn(Motion::at(Transform::IDENTITY, 1.0));
    world
}

/// 物理刻调度：固定步长跑这个。
pub fn tick_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(apply_input_movement);
    schedule
}

/// 渲染刻调度：按帧跑这个。
pub fn render_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(collect_world_matrices);
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_timestep_accumulates_ticks() {
        let step = Duration::from_secs_f64(1.0 / 60.0);
        let mut ts = FixedTimestep::new(step);
        // 2.5 步 → 2 个 tick，alpha ≈ 0.5
        let (ticks, alpha) = ts.advance(step.mul_f64(2.5));
        assert_eq!(ticks, 2);
        assert!((alpha - 0.5).abs() < 1e-3);
        // 不足一步 → 0 tick
        let (ticks, _) = ts.advance(step.mul_f64(0.2));
        assert_eq!(ticks, 0);
    }

    #[test]
    fn tick_schedule_moves_entity_at_fixed_step() {
        let mut world = demo_world();
        let mut schedule = tick_schedule();
        world.resource_mut::<InputSnapshot>().move_dir = Vec3::new(1.0, 0.0, 0.0);
        // 60 个物理刻 = 1 秒：位置应精确推进 1 单位。
        for _ in 0..60 {
            schedule.run(&mut world);
        }
        let motion = world.query::<&Motion>().iter(&world).next().expect("演示实体");
        assert!((motion.transform.position.x - 1.0).abs() < 1e-5);
        assert_eq!(motion.transform.position.y, 0.0);
    }

    #[test]
    fn render_schedule_collects_world_matrices() {
        let mut world = demo_world();
        let mut ticks = tick_schedule();
        let mut renders = render_schedule();
        world.resource_mut::<InputSnapshot>().move_dir = Vec3::new(0.0, 0.0, 1.0);
        ticks.run(&mut world); // 1 个物理刻
        renders.run(&mut world); // 1 帧渲染消费
        let matrices = &world.resource::<RenderedMatrices>().0;
        assert_eq!(matrices.len(), 1);
        let (_, _, translation) = matrices[0].to_scale_rotation_translation();
        assert!((translation.z - 1.0 / 60.0).abs() < 1e-5);
    }
}
