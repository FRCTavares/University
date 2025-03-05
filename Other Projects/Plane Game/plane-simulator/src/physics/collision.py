def check_collision(entity1, entity2):
    # Simple AABB collision detection
    return (entity1.position.x < entity2.position.x + entity2.size.x and
            entity1.position.x + entity1.size.x > entity2.position.x and
            entity1.position.y < entity2.position.y + entity2.size.y and
            entity1.position.y + entity1.size.y > entity2.position.y and
            entity1.position.z < entity2.position.z + entity2.size.z and
            entity1.position.z + entity1.size.z > entity2.position.z)

def handle_collision(entity1, entity2):
    if check_collision(entity1, entity2):
        # Simple response: move entity1 out of entity2
        overlap_x = (entity1.position.x + entity1.size.x / 2) - (entity2.position.x + entity2.size.x / 2)
        overlap_y = (entity1.position.y + entity1.size.y / 2) - (entity2.position.y + entity2.size.y / 2)
        overlap_z = (entity1.position.z + entity1.size.z / 2) - (entity2.position.z + entity2.size.z / 2)

        if abs(overlap_x) > abs(overlap_y) and abs(overlap_x) > abs(overlap_z):
            entity1.position.x -= overlap_x
        elif abs(overlap_y) > abs(overlap_x) and abs(overlap_y) > abs(overlap_z):
            entity1.position.y -= overlap_y
        else:
            entity1.position.z -= overlap_z

def update_collisions(entities):
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            handle_collision(entities[i], entities[j])