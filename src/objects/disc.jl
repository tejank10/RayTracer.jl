# -------- #
# - Disc - #
# -------- #

# TODO: Use barycentric coordinates and Moller-Trumbore Algorithm
mutable struct Disc{V,T<:Real} <: Object
    center::Vec3{V}
    normal::Vec3{V}
    radius::T
    material::Material
end 

@diffops Disc

# The next 3 functions are just convenience functions for handling
# gradients properly for getproperty function
function Disc(v::Vec3{T}, sym::Symbol) where {T}
    z = eltype(T)(0)
    mat = Material(PlainColor(rgb(z)), z)
    if sym == :center
        return Disc(v, Vec3(z), z, mat)
    else
        return Disc(Vec3(z), v, z, mat)
    end
end

function Disc(r::T, ::Symbol) where {T<:Real}
    z = T(0)
    mat = Material(PlainColor(rgb(z)), z)
    return Disc(Vec3(z), Vec3(z), r, mat)
end

# Don't take gradients for material
function Disc(mat::Material{S, R}, ::Symbol) where {S, R}
    z = R(0)
    mat2 = Material(PlainColor(rgb(z)), z)
    return Disc(Vec3(z), Vec3(z), z, mat2)
end

function Disc(n::Vec3, r::T, c::Vec3; color = rgb(0.5f0), reflection = 0.5f0) where {T<:Real}
    mat = Material(PlainColor(color), reflection)
    n = normalize(n)
    return Disc(c, n, r, mat)
end

function intersect(d::Disc, origin, direction)
    dot_dn = dot(direction, d.normal)
    p_org = d.center - origin
    t = dot(p_org, d.normal) ./ dot_dn
    pt = origin + direction * t
    dist = l2norm(pt - d.center)
    r2 = d.radius ^ 2
    function get_intersection(t₀, dst)
        t_ = t₀
        if t₀ < 0 || dst > r2
            t_ = bigmul(t₀ + dst + r2)
        end
        return t_
    end
    result = broadcast(get_intersection, t, dist)
    return result
end

function get_normal(t::Disc, pt, direction)
    normal_dir = dot(t.normal, direction)
    mult_factor = broadcast(x -> x < 0 ? one(typeof(x)) : -one(typeof(x)), normal_dir)
    normal = Vec3(repeat(t.normal.x, inner = size(pt.x)),
                  repeat(t.normal.y, inner = size(pt.y)),
                  repeat(t.normal.z, inner = size(pt.z)))
    return normal * mult_factor
end