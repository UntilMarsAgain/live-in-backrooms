//! 输入动作（Input Action）体系：把物理按键映射成语义动作，各系统按动作响应。
//!
//! 层次：
//! - 原始层：[`super::InputSnapshot`]——winit 事件回调只写按键状态/鼠标增量；
//! - 动作层：本模块——[`bind_input`] 系统（物理刻开头）按 [`InputBindings`]
//!   把按键映射成 [`ActionState`]（持续按住）与 [`ActionEvents`]（本刻
//!   刚按下/释放）。
//!
//! 好处：系统/物体只关心"动作"（MoveForward / Interact…），不关心物理键；
//! 键位可重绑（改 [`InputBindings`] 即可），新输入行为 = 加一个系统订阅动作，
//! 不用改 App 的分发。

use std::collections::HashSet;

use bevy_ecs::prelude::*;
use winit::keyboard::KeyCode;

use crate::engine::ecs::InputSnapshot;

/// 语义输入动作：与物理键解耦，绑定表决定"哪个键触发哪个动作"。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputAction {
    // 移动 / 视角（自由相机）。
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    // 调试可视化开关。
    ToggleLightDebug,
    ToggleCollisionDebug,
    // 演示场景切换（App 级副作用，App 查询事件后响应）。
    SwitchDemo1,
    SwitchDemo2,
    // 释放鼠标捕获。
    ReleaseMouse,
    /// 占位：未来实体交互（拾取、开关）用。
    Interact,
}

/// 动作 → 按键的绑定表（一个动作可绑多个键，如 W / ↑ 都是向前）。
///
/// 默认绑定在 [`InputBindings::default`]；需要重绑/加自定义动作时改这个
/// 资源（未来可来自配置文件）。
#[derive(Resource, Debug, Clone)]
pub struct InputBindings {
    map: Vec<(InputAction, Vec<KeyCode>)>,
}

impl Default for InputBindings {
    fn default() -> Self {
        Self {
            map: vec![
                (
                    InputAction::MoveForward,
                    vec![KeyCode::KeyW, KeyCode::ArrowUp],
                ),
                (
                    InputAction::MoveBackward,
                    vec![KeyCode::KeyS, KeyCode::ArrowDown],
                ),
                (
                    InputAction::MoveLeft,
                    vec![KeyCode::KeyA, KeyCode::ArrowLeft],
                ),
                (
                    InputAction::MoveRight,
                    vec![KeyCode::KeyD, KeyCode::ArrowRight],
                ),
                (InputAction::MoveUp, vec![KeyCode::Space]),
                (InputAction::MoveDown, vec![KeyCode::ShiftLeft]),
                (InputAction::ToggleLightDebug, vec![KeyCode::KeyL]),
                (InputAction::ToggleCollisionDebug, vec![KeyCode::KeyB]),
                (InputAction::SwitchDemo1, vec![KeyCode::F1]),
                (InputAction::SwitchDemo2, vec![KeyCode::F2]),
                (InputAction::ReleaseMouse, vec![KeyCode::Escape]),
                (InputAction::Interact, vec![KeyCode::KeyE]),
            ],
        }
    }
}

impl InputBindings {
    /// 查询一个动作绑定的所有按键（None = 未绑定）。
    pub fn keys_for(&self, action: InputAction) -> Option<&[KeyCode]> {
        self.map
            .iter()
            .find(|(a, _)| *a == action)
            .map(|(_, keys)| keys.as_slice())
    }

    /// 查询一个按键对应的动作（未绑定返回 None）。
    pub fn action_for(&self, code: KeyCode) -> Option<InputAction> {
        self.map
            .iter()
            .find(|(_, keys)| keys.contains(&code))
            .map(|(action, _)| *action)
    }

    /// 给动作绑定一个键（加到该动作的键列表；动作未登记时新建条目）。
    pub fn bind(&mut self, action: InputAction, code: KeyCode) {
        if let Some((_, keys)) = self.map.iter_mut().find(|(a, _)| *a == action) {
            if !keys.contains(&code) {
                keys.push(code);
            }
        } else {
            self.map.push((action, vec![code]));
        }
    }
}

/// 当前**按住**的动作集合（持续状态：MoveForward 等按住移动用）。
#[derive(Resource, Debug, Default)]
pub struct ActionState {
    pressed: HashSet<InputAction>,
}

impl ActionState {
    pub fn pressed(&self, action: InputAction) -> bool {
        self.pressed.contains(&action)
    }
}

/// 本物理刻**刚按下 / 刚释放**的动作（边沿事件：开关、交互用）。
///
/// 由 [`bind_input`] 消费后清空，避免跨刻重复触发。
#[derive(Resource, Debug, Default)]
pub struct ActionEvents {
    just_pressed: HashSet<InputAction>,
    just_released: HashSet<InputAction>,
}

impl ActionEvents {
    pub fn just_pressed(&self, action: InputAction) -> bool {
        self.just_pressed.contains(&action)
    }

    pub fn just_released(&self, action: InputAction) -> bool {
        self.just_released.contains(&action)
    }

    fn clear(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }
}

/// 物理刻开头的输入绑定系统：按键快照 → 动作状态 + 边沿事件。
///
/// 用 `Local` 记住上一刻按键做差分：本刻新增 = just_pressed，本刻消失 =
/// just_released。动作状态 = 本刻按下的键对应的动作集合。
pub fn bind_input(
    input: Res<InputSnapshot>,
    bindings: Res<InputBindings>,
    mut state: ResMut<ActionState>,
    mut events: ResMut<ActionEvents>,
    mut prev_keys: Local<HashSet<KeyCode>>,
) {
    // 边沿：相对上一刻的差分。
    let pressed_now: HashSet<InputAction> = input
        .keys
        .iter()
        .filter_map(|code| bindings.action_for(*code))
        .collect();
    let prev_actions: HashSet<InputAction> = prev_keys
        .iter()
        .filter_map(|code| bindings.action_for(*code))
        .collect();

    events.just_pressed = pressed_now.difference(&prev_actions).copied().collect();
    events.just_released = prev_actions.difference(&pressed_now).copied().collect();
    state.pressed = pressed_now;

    // 下一刻的差分基准。
    *prev_keys = input.keys.clone();
}

#[cfg(test)]
mod tests {
    use bevy_ecs::prelude::*;

    use super::*;

    fn world_with_input(keys: &[KeyCode]) -> (World, Schedule) {
        let mut world = World::new();
        world.insert_resource(InputBindings::default());
        world.insert_resource(ActionState::default());
        world.insert_resource(ActionEvents::default());
        world.insert_resource(InputSnapshot {
            keys: keys.iter().copied().collect(),
            ..Default::default()
        });
        let mut schedule = Schedule::default();
        schedule.add_systems(bind_input);
        (world, schedule)
    }

    /// 按键 → 动作状态 + 边沿事件：刚按下出现 just_pressed，按住保持 pressed。
    #[test]
    fn bind_input_maps_keys_to_actions_and_edges() {
        let (mut world, mut schedule) = world_with_input(&[KeyCode::KeyW, KeyCode::KeyL]);
        schedule.run(&mut world);

        let state = world.resource::<ActionState>();
        assert!(state.pressed(InputAction::MoveForward), "W → MoveForward");
        assert!(!state.pressed(InputAction::MoveBackward));
        let events = world.resource::<ActionEvents>();
        assert!(
            events.just_pressed(InputAction::MoveForward),
            "首次按下产生边沿"
        );
        assert!(
            events.just_pressed(InputAction::ToggleLightDebug),
            "L → 调试开关"
        );
    }

    /// 边沿只在按键变化那一刻出现：连续两刻按住 W，第二刻不再是 just_pressed。
    #[test]
    fn edges_only_fire_on_change() {
        let (mut world, mut schedule) = world_with_input(&[KeyCode::KeyW]);
        schedule.run(&mut world);
        assert!(
            world
                .resource::<ActionEvents>()
                .just_pressed(InputAction::MoveForward)
        );

        // 第二刻仍按住 W：状态保持，但不再有 just_pressed 边沿。
        schedule.run(&mut world);
        let state = world.resource::<ActionState>();
        assert!(state.pressed(InputAction::MoveForward));
        let events = world.resource::<ActionEvents>();
        assert!(!events.just_pressed(InputAction::MoveForward));

        // 松开 W：出现 just_released。
        world.resource_mut::<InputSnapshot>().keys.clear();
        schedule.run(&mut world);
        assert!(
            world
                .resource::<ActionEvents>()
                .just_released(InputAction::MoveForward)
        );
        assert!(
            !world
                .resource::<ActionState>()
                .pressed(InputAction::MoveForward)
        );
    }
}
