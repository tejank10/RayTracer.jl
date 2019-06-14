module RayTracer

<<<<<<< HEAD
using Zygote, Flux, Images, FileIO, MeshIO, CUDAnative, CuArrays
using Flux: @treelike
using Zygote: @adjoint
=======
using Zygote, Flux, Images
import Base.show
>>>>>>> e61147c1abbb0eb010d3b5954a49c81fef7abc0a

# Rendering Utilities
include("utils.jl")
include("light.jl")
include("materials.jl")
include("objects.jl")
include("camera.jl")
include("optimize.jl")

# Renderers
include("renderers/blinnphong.jl")

# Image Utilities
include("imutils.jl")

# Differentiable Rendering
include("gradients/zygote.jl")
include("gradients/numerical.jl")

end
