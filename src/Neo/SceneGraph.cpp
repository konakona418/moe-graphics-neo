#include "Neo/SceneGraph.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace moe::neo {
    glm::mat4 Node::LocalTransform() const {
        const glm::mat4 translation = glm::translate(glm::mat4(1.0f), mTranslation);
        const glm::mat4 rotation = glm::mat4_cast(mRotation);
        const glm::mat4 scale = glm::scale(glm::mat4(1.0f), mScale);
        return translation * rotation * scale;
    }

    std::vector<glm::mat4> SceneGraph::ComputeWorldTransforms() const {
        std::vector<glm::mat4> world(mNodes.size(), glm::mat4(1.0f));

        moe::SmallVector<uint32_t, 8> stack = mRootNodes;
        while (!stack.empty()) {
            const uint32_t index = stack.back();
            stack.pop_back();
            const Node& node = mNodes[index];
            const glm::mat4 local = node.LocalTransform();
            world[index] = node.mParent >= 0 ? world[node.mParent] * local : local;
            for (const uint32_t child : node.mChildren) {
                stack.push_back(child);
            }
        }
        return world;
    }
}// namespace moe::neo