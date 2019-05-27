export raytrace

# ----------------- #
# Blinn Phong Model #
# ----------------- #

"""
    light(s::Object, origin, direction, dist, lgt::Light, eye_pos, scene, obj_num, bounce)

Implements the Blinn Phong rendering algorithm. This function is merely for
internal usage and should in no case be called by the user. This function is
quite general and supports user defined **Objects**. For support of custom
Objects have a look at the examples.
"""
function light(s::S, origin, direction, dist, lgt::L, eye_pos,
               scene, obj_num, bounce) where {S<:Object, L<:Light}
    pt = origin + direction * dist
    normal = get_normal(s, pt, direction)
    dir_light, intensity = get_shading_info(lgt, pt)
    dir_origin = normalize(eye_pos - pt)
    nudged = pt + normal * 0.0001f0 # Nudged to miss itself
    
    # Shadow
    light_distances = broadcast(x -> intersect(x, nudged, dir_light), scene)
    seelight = fseelight(obj_num, light_distances)
    
    # Ambient
    color = rgb(0.05f0)

    # Lambert Shading (diffuse)
    visibility = max.(dot(normal, dir_light), 0.0f0)
    color += ((diffusecolor(s, pt) * intensity) * visibility) * seelight
    
    # Reflection
    if bounce < 2
        rayD = normalize(direction - normal * 2.0f0 * dot(direction, normal))
        color += raytrace(nudged, rayD, scene, lgt, eye_pos, bounce + 1) *
                 s.material.reflection
    end

    # Blinn-Phong shading (specular)
    phong = dot(normal, normalize(dir_light + dir_origin))
    color += (rgb(1.0f0) * (clamp.(phong, 0.0f0, 1.0f0) .^ 50)) * seelight

    return color
end

"""
    raytrace(origin::Vec3, direction::Vec3, scene::Vector, lgt::Light, eye_pos::Vec3, bounce::Int)
    raytrace(origin::Vec3, direction::Vec3, scene::Vector, lgt::Vector{Light}, eye_pos::Vec3, bounce::Int)

Computes the color contribution to every pixel by tracing every single ray.
Internally it calls the `light` function which implements Blinn Phong Rendering
and adds up the color contribution for each object.

The `eye_pos` is simply the `origin` when called by the user. However, the origin
keeps changing across the recursive calls to this function and hence it is
necessary to keep track of the `eye_pos` separately.

The `bounce` parameter allows the configuration of global illumination. To turn off
global illumination set the `bounce` parameter to `>= 2`. As expected rendering is
much faster if global illumination is off but at the same time is much less photorealistic.

!!! note
    The support for multiple lights is primitive as we loop over the lights.
    Even though it is done in a parallel fashion, it is not the best way to
    do so. Nevertheless it exists just for the sake of experimentation.
"""
function raytrace(origin::Vec3, direction::Vec3, scene::Vector,
                  lgt::L, eye_pos::Vec3, bounce::Int = 0) where {L<:Light}
    distances = broadcast(x -> intersect(x, origin, direction), scene)

    nearest = map(min, distances...)
    h = isnotbigmul.(nearest)

    color = rgb(0.0f0)

    for (i, (s, d)) in enumerate(zip(scene, distances))
        hit = hashit.(h, d, nearest)
        if sum(hit) != 0
            dc = extract(hit, d)
            originc = extract(hit, origin)
            dirc = extract(hit, direction)
            cc = light(s, originc, dirc, dc, lgt, eye_pos, scene, i, bounce)
            color += place(cc, hit)
        end
    end

    return color
end

# FIXME: This is a temporary solution. Long term solution is to support
#        a light vector in the `light` function itself.
function raytrace(origin::Vec3, direction::Vec3, scene::Vector,
                  lgt::Vector{L}, eye_pos::Vec3, bounce::Int = 0) where {L<:Light}
    colors = pmap(x -> raytrace(origin, direction, scene, x, eye_pos, 0), lgt)
    return sum(colors)
end

fseelight(n, light_distances) = map((x...) -> min(x...) == x[n], light_distances...)
