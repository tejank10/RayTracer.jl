export Camera, get_primary_rays

# ------ #
# Camera #
# ------ #

struct FixedCameraParams{T} <: FixedParams
    vup::Vec3{T}
    width
    height
end

@treelike FixedCameraParams

# Incorporate `aperture` later
mutable struct Camera{T}
    lookfrom::Vec3{T}
    lookat::Vec3{T}
    vfov::T
    focus::T
    fixedparams::FixedCameraParams{T}
end

@treelike Camera
                                       
@diffops Camera

function Camera(lookfrom, lookat, vup, vfov, focus, width, height)
    fixedparams = FixedCameraParams(vup, width, height)
    return Camera(lookfrom, lookat, [vfov], [focus], fixedparams)
end

"""
    get_primary_rays(c::Camera)

Takes the configuration of the camera and returns
the origin and the direction of the primary rays.
"""
# We assume that the camera is at a unit distance from the screen
function get_primary_rays(c::Camera)
    width = c.fixedparams.width
    height = c.fixedparams.height
    vup = c.fixedparams.vup
    vfov = c.vfov[]
    focus = c.focus[]

    aspect_ratio = width / height
    half_height = tan(deg2rad(vfov / 2))
    half_width = typeof(half_height)(aspect_ratio * half_height)

    origin = c.lookfrom
    w = normalize(c.lookfrom - c.lookat)
    u = normalize(cross(w, vup))
    v = normalize(cross(w, u))

    # Lower Left Corner
    llc = origin - half_width * focus * u - half_height * focus * v - w
    hori = 2 * half_width * focus * u
    vert = 2 * half_height * focus * v

    s = repeat((collect(0:(width - 1)) .+ 0.5f0) ./ width, outer = height) |> gpu
    t = repeat((collect(0:(height - 1)) .+ 0.5f0) ./ height, inner = width) |> gpu
    
    direction = normalize(llc + s * hori + t * vert - origin)

    return (origin, direction)
end
