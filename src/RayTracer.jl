module RayTracer

using Zygote, Flux, Images, FileIO, MeshIO, CUDAnative
using Flux: @treelike
using Zygote: @adjoint

# Rendering Utilities
include("utils.jl")
include("light.jl")
include("materials.jl")
include("objects.jl")
include("camera.jl")
include("optimize.jl")

# Renderers
include("renderers/blinnphong.jl")

# Differentiable Rendering
include("gradients/zygote.jl")
include("gradients/numerical.jl")

# Image Utilities
include("imutils.jl")

end
