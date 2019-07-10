export rasterize

# --------- #
# Constants #
# --------- #

const film_aperture = (0.980f0, 0.735f0)

# --------- #
# Utilities #
# --------- #

edge_function(pt1::Vec3, pt2::Vec3, point::Vec3) = edge_function_vector(pt1, pt2, point)[]

edge_function_vector(pt1::Vec3, pt2::Vec3, point::Vec3) =
    ((point.x .- pt1.x) .* (pt2.y - pt1.y) .- (point.y .- pt1.y) .* (pt2.x .- pt1.x))

function convert2raster(vertex_world::Vec3, world_to_camera, left::Real, right::Real,
                        top::Real, bottom::Real, width::Int, height::Int)
    vertex_camera = world2camera(vertex_world, world_to_camera)

    return convert2raster(vertex_camera, left, right, top, bottom, width, height)
end

function convert2raster(vertex_camera::Vec3{T}, left::Real, right::Real, top::Real, bottom::Real,
                        width::Int, height::Int) where {T}
    outtype = eltype(T)

    vertex_screen = (x = vertex_camera.x[] / -vertex_camera.z[],
                     y = vertex_camera.y[] / -vertex_camera.z[])

    vertex_NDC = (x = outtype((2 * vertex_screen.x - right - left) / (right - left)),
                  y = outtype((2 * vertex_screen.y - top - bottom) / (top - bottom)))

    vertex_raster = Vec3([(vertex_NDC.x + 1) / 2 * outtype(width)],
                         [(1 - vertex_NDC.y) / 2 * outtype(height)],
                         -vertex_camera.z)

    return vertex_raster
end

# ---------- #
# Rasterizer #
# ---------- #

function rasterize(cam::Camera, scene::Vector)
    top, right, bottom, left = compute_screen_coordinates(cam, film_aperture)
    camera_to_world = get_transformation_matrix(cam)
    world_to_camera = inv(camera_to_world)
    return rasterize(cam, scene, camera_to_world, world_to_camera, top,
                     right, bottom, left)
end

function rasterize(cam::Camera{T}, scene::Vector, camera_to_world,
                   world_to_camera, top, right, bottom, left) where {T}
    width = cam.fixedparams.width
    height = cam.fixedparams.height

    frame_buffer = Vec3(zeros(eltype(T), width * height))
    depth_buffer = fill(eltype(T)(Inf), width * height)

    for triangle in scene
        v1_camera = world2camera(triangle.v1, world_to_camera)
        v2_camera = world2camera(triangle.v2, world_to_camera)
        v3_camera = world2camera(triangle.v3, world_to_camera)

        normal = normalize(cross(v2_camera - v1_camera, v3_camera - v1_camera))

        v1_raster = convert2raster(v1_camera, left, right, top, bottom, width, height)
        v2_raster = convert2raster(v2_camera, left, right, top, bottom, width, height)
        v3_raster = convert2raster(v3_camera, left, right, top, bottom, width, height)

        # Bounding Box
        xmin, xmax = extrema([v1_raster.x[], v2_raster.x[], v3_raster.x[]])
        ymin, ymax = extrema([v1_raster.y[], v2_raster.y[], v3_raster.y[]])

        (xmin > width-1 || xmax < 0 || ymin > height-1 || ymax < 0) && continue

        area = edge_function(v1_raster, v2_raster, v3_raster)

        # Loop over only the covered pixels
        x₁ = max(       0, Int(floor(xmin)))
        x₂ = min( width-1, Int(floor(xmax)))
        y₁ = max(       0, Int(floor(ymin)))
        y₂ = min(height-1, Int(floor(ymax)))

        y = y₁:y₂
        x = x₁:x₂

        y_space = repeat(collect(y), inner = length(x))
        x_space = repeat(collect(x), outer = length(y))

        w1_arr = Float32[]
        w2_arr = Float32[]
        w3_arr = Float32[]
        depth  = Float32[]
        x_arr  = Int[]
        y_arr  = Int[]

        y_vec = y_space .+ 0.5f0
        x_vec = x_space .+ 0.5f0

        pixel = Vec3(x_vec, y_vec, zeros(eltype(x_vec), length(x) * length(y)))
        w1 = edge_function_vector(v2_raster, v3_raster, pixel)
        w2 = edge_function_vector(v3_raster, v1_raster, pixel)
        w3 = edge_function_vector(v1_raster, v2_raster, pixel)

        for (w1_val, w2_val, w3_val, x_val, y_val) in zip(w1, w2, w3, x_space, y_space)
            if w1_val >= 0 && w2_val >= 0 && w3_val >= 0
                w1_val = w1_val / area
                w2_val = w2_val / area
                w3_val = w3_val / area

                depth_val = 1 / (w1_val / v1_raster.z[] + w2_val / v2_raster.z[] +
                                 w3_val / v3_raster.z[])

                if depth_val < depth_buffer[y_val*width+x_val+1]
                    update_index!(depth_buffer, y_val*width+x_val+1, depth_val)

                    px = (v1_camera.x[] / -v1_camera.z[]) .* w1_val .+
                         (v2_camera.x[] / -v2_camera.z[]) .* w2_val .+
                         (v3_camera.x[] / -v3_camera.z[]) .* w3_val

                    py = (v1_camera.y[] / -v1_camera.z[]) .* w1_val .+
                         (v2_camera.y[] / -v2_camera.z[]) .* w2_val .+
                         (v3_camera.y[] / -v3_camera.z[]) .* w3_val

                    # Passing these gradients as 1.0f0 is incorrect
                    pt = Zygote.hook(Δ -> Vec3([1.0f0 for _ in pt.x]),
                                     camera2world(Vec3(px*depth_val, py*depth_val, -depth_val),
                                                  camera_to_world))

                    col = get_color(triangle, pt, Val(:diffuse))
                    idx = y_val .* width .+ x_val .+ 1

                    frame_buffer = place_idx!(frame_buffer, col, idx)
                end
            end
        end
    end

    return frame_buffer
end
