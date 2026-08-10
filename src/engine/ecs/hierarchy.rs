//! 层级：bevy 内置 `ChildOf` / `Children` 关系 + 世界矩阵传播系统。

use std::collections::HashSet;

use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;

use super::components::{LocalTransform, WorldMatrix};

/// 世界矩阵传播：从根（无 `ChildOf`）出发按层序收集（父先于子），
/// 逐实体累乘父世界矩阵 × 局部矩阵。
///
/// 层序 + 访问集合保证成环也能终止（成环节点的后代按"已访问"跳过）。
pub fn propagate_world_transforms(
    nodes: Query<(Entity, &LocalTransform, Option<&ChildOf>)>,
    mut worlds: Query<&mut WorldMatrix>,
    children: Query<&Children>,
) {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut queue: Vec<Entity> = nodes
        .iter()
        .filter(|(_, _, child_of)| child_of.is_none())
        .map(|(entity, _, _)| entity)
        .collect();
    while let Some(entity) = queue.pop() {
        if !visited.insert(entity) {
            continue;
        }
        order.push(entity);
        if let Ok(child_list) = children.get(entity) {
            queue.extend((&**child_list).iter().copied());
        }
    }

    for entity in order {
        let Ok((_, local, child_of)) = nodes.get(entity) else {
            continue;
        };
        let local_mat = local.0.to_mat4();
        let world = match child_of {
            Some(child_of) => worlds
                .get(child_of.0)
                .map(|parent_world| parent_world.0 * local_mat)
                .unwrap_or(local_mat),
            None => local_mat,
        };
        if let Ok(mut world_matrix) = worlds.get_mut(entity) {
            world_matrix.0 = world;
        }
    }
}
