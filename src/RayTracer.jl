module RayTracer

using Zygote, Flux, Images, Distributed, CUDAnative, CuArrays
using Flux: @treelike
using Zygote: @adjoint
import Base.show

# Rendering Utilities
include("utils.jl")
include("light.jl")
include("materials.jl")
include("objects.jl")
include("camera.jl")
include("optimize.jl")

# Renderers
include("renderers/blinnphong.jl")
include("renderers/rasterizer.jl")

# Image Utilities
include("imutils.jl")

# Differentiable Rendering
include("gradients/zygote.jl")
include("gradients/numerical.jl")

end
