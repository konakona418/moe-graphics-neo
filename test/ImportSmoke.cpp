#include <Neo/Importer.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond)                                            \
    do {                                                       \
        if (!(cond)) {                                         \
            std::fprintf(stderr, "Import FAILED: %s (%d)\n",   \
                    #cond, __LINE__);                          \
            return EXIT_FAILURE;                               \
        }                                                      \
    } while (false)

int main() {
    moe::neo::Scene scene;
    std::string error;
    if (!moe::neo::ImportGltf(MOE_SOURCE_DIR "/test/assets/minimal.gltf", scene, error)) {
        std::fprintf(stderr, "Import FAILED: %s\n", error.c_str());
        return EXIT_FAILURE;
    }

    // node hierarchy preserved (empty root node + child with mesh)
    CHECK(scene.mGraph.mNodes.size() == 2);
    CHECK(scene.mGraph.mNodes[0].mName == "root");
    CHECK(scene.mGraph.mNodes[0].mParent == -1);
    CHECK(scene.mGraph.mNodes[0].mChildren.size() == 1);
    CHECK(scene.mGraph.mNodes[0].mChildren[0] == 1);
    CHECK(scene.mGraph.mNodes[1].mName == "triangle");
    CHECK(scene.mGraph.mNodes[1].mParent == 0);
    CHECK(scene.mGraph.mNodes[1].mMeshes.size() == 1);
    CHECK(scene.mGraph.mNodes[1].mMeshes[0] == 0);

    // TRS preserved
    CHECK(scene.mGraph.mNodes[0].mTranslation == glm::vec3(1.0f, 0.0f, 0.0f));
    CHECK(scene.mGraph.mNodes[1].mScale == glm::vec3(1.0f, 1.0f, 1.0f));

    // scene roots
    CHECK(scene.mSceneRoots.size() == 1);
    CHECK(scene.mSceneRoots[0] == 0);

    // mesh
    CHECK(scene.mMeshes.size() == 1);
    CHECK(scene.mMeshes[0].mName == "tri");
    CHECK(scene.mMeshes[0].mPrimitives.size() == 1);
    const auto& prim = scene.mMeshes[0].mPrimitives[0];
    CHECK(prim.mPositions.size() == 3);
    CHECK(prim.mPositions[1] == glm::vec3(1.0f, 0.0f, 0.0f));
    CHECK(prim.mIndices.size() == 3);
    CHECK(prim.mIndices[0] == 0 && prim.mIndices[1] == 1 && prim.mIndices[2] == 2);
    CHECK(prim.mMaterialIndex == 0);

    // material
    CHECK(scene.mMaterials.size() == 1);
    CHECK(scene.mMaterials[0].mBaseColor == glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    CHECK(scene.mMaterials[0].mRoughness == 1.0f);

    // world transform: node1 world = root.translation * identity = translate(1,0,0)
    const auto world = scene.mGraph.ComputeWorldTransforms();
    CHECK(world[1] == glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

    std::printf("Import smoke passed.\n");
    return EXIT_SUCCESS;
}