//! 层级：`Parent` / `Children` 组件 + 世界矩阵传播系统。

use std::collections::HashSet;

use bevy_ecs::prelude::*;

use super::components::{Children, LocalTransform, Parent, WorldMatrix};

/// 世界矩阵传播：从根（无 `Parent`）出发按层序收集（父先于子），
/// 逐实体累乘父世界矩阵 × 局部矩阵。
///
/// 层序 + 访问集合保证成环也能终止（成环节点的后代按"已访问"跳过）。
pub fn propagate_world_transforms(
    nodes: Query<(Entity, &LocalTransform, Option<&Parent>)>,
    mut worlds: Query<&mut WorldMatrix>,
    children: Query<&Children>,
) {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut queue: Vec<Entity> = nodes
        .iter()
        .filter(|(_, _, parent)| parent.is_none())
        .map(|(entity, _, _)| entity)
        .collect();
    while let Some(entity) = queue.pop() {
        if !visited.insert(entity) {
            continue;
        }
        order.push(entity);
        if let Ok(children) = children.get(entity) {
            queue.extend(children.0.iter().copied());
        }
    }

    for entity in order {
        let Ok((_, local, parent)) = nodes.get(entity) else {
            continue;
        };
        let local_mat = local.0.to_mat4();
        let world = match parent {
            Some(parent) => worlds
                .get(parent.0)
                .map(|parent_world| parent_world.0 * local_mat)
                .unwrap_or(local_mat),
            None => local_mat,
        };
        if let Ok(mut world_matrix) = worlds.get_mut(entity) {
            world_matrix.0 = world;
        }
    }
}
