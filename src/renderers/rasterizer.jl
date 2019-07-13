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
                        top::Real, bottom::Real, near::Real, width::Int, height::Int)
    vertex_camera = world2camera(vertex_world, world_to_camera)

    return convert2raster(vertex_camera, left, right, top, bottom, near, width, height)
end

function convert2raster(vertex_camera::Vec3{T}, left::Real, right::Real, top::Real, bottom::Real,
                        near::Real, width::Int, height::Int) where {T}
    outtype = eltype(T)

    vertex_screen = (x = near * vertex_camera.x[] / -vertex_camera.z[],
                     y = near * vertex_camera.y[] / -vertex_camera.z[])

    vertex_NDC = (x = outtype((2f0 * vertex_screen.x - right - left) / (right - left)),
                  y = outtype((2f0 * vertex_screen.y - top - bottom) / (top - bottom)))

    vertex_raster = Vec3([(vertex_NDC.x + 1f0) / 2f0 * outtype(width)],
                         [(1f0 - vertex_NDC.y) / 2f0 * outtype(height)],
                         -vertex_camera.z)

    return vertex_raster
end

# ---------- #
# Rasterizer #
# ---------- #

function rasterize(cam::Camera, scene::Vector,
                   near_clipping_plane::Real=1f0, far_clipping_plane::Real=Float32(Inf))
    top, right, bottom, left = compute_screen_coordinates(cam, film_aperture, near_clipping_plane)
    camera_to_world = get_transformation_matrix(cam)
    world_to_camera = inv(camera_to_world)
    return rasterize(cam, scene, near_clipping_plane, far_clipping_plane,
                     camera_to_world, world_to_camera, top, right, bottom, left)
end

function rasterize(cam::Camera{T}, scene::Vector, near_clipping_plane,
                   far_clipping_plane, camera_to_world, world_to_camera,
                   top, right, bottom, left) where {T}
    width = cam.fixedparams.width
    height = cam.fixedparams.height

    frame_buffer = Vec3(zeros(eltype(T), width * height))
    depth_buffer = fill(eltype(T)(far_clipping_plane), width * height)

    for triangle in scene
        v1_camera = world2camera(triangle.v1, world_to_camera)
        v2_camera = world2camera(triangle.v2, world_to_camera)
        v3_camera = world2camera(triangle.v3, world_to_camera)

        normal = normalize(cross(v2_camera - v1_camera, v3_camera - v1_camera))

        v1_raster = convert2raster(v1_camera, left, right, top, bottom, near_clipping_plane, width, height)
        v2_raster = convert2raster(v2_camera, left, right, top, bottom, near_clipping_plane, width, height)
        v3_raster = convert2raster(v3_camera, left, right, top, bottom, near_clipping_plane, width, height)

        # Bounding Box
        xmin, xmax = extrema([v1_raster.x[], v2_raster.x[], v3_raster.x[]])
        ymin, ymax = extrema([v1_raster.y[], v2_raster.y[], v3_raster.y[]])

        (xmin > width-1 || xmax < 0 || ymin > height-1 || ymax < 0) && continue

        area = edge_function(v1_raster, v2_raster, v3_raster)

        # Loop over only the covered pixels
        x₁ = max(     1, 1+Int(floor(xmin)))
        x₂ = min( width, 1+Int(floor(xmax)))
        y₁ = max(     1, 1+Int(floor(ymin)))
        y₂ = min(height, 1+Int(floor(ymax)))

        y = y₁:y₂
        x = x₁:x₂
        for y_val in y₁:y₂
            for x_val in x₁:x₂
                pixel = Vec3([x_val+0.5f0], [y_val+0.5f0], [0f0])
                w1_val = edge_function(v2_raster, v3_raster, pixel)
                w2_val = edge_function(v3_raster, v1_raster, pixel)
                w3_val = edge_function(v1_raster, v2_raster, pixel)

                if w1_val >= 0 && w2_val >= 0 && w3_val >= 0
                    w1_val = w1_val / area
                    w2_val = w2_val / area
                    w3_val = w3_val / area

                    depth_val = 1 / (w1_val / v1_raster.z[] + w2_val / v2_raster.z[] +
                                     w3_val / v3_raster.z[])

                    if depth_val < depth_buffer[(y_val-1)*width+x_val]
                        update_index!(depth_buffer, (y_val-1)*width+x_val, depth_val)

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
                        idx = (y_val-1)*width+x_val

                        frame_buffer = place_idx!(frame_buffer, col, idx)
                    end
                end
            end
        end
    end
    
    return Vec3(min.(0f0, depth_buffer) ./ far_clipping_plane)
end
