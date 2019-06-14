using RayTracer, Images, Flux, Zygote, Stattistics
using RayTracer: improcess

screen_size = (w = 128, h = 128)

light = DistantLight(Vec3(1.0f0), 100.0f0, Vec3(0.0f0, 1.0f0, 0.0f0))

scene = [
    # Earth Surface
    SimpleSphere(Vec3(0.0f0, -99999.5f0, 0.0f0), 99999.0f0, color = Vec3(0.0f0, 1.0f0, 0.0f0)),
    # Road defined by 2 triangles
    Triangle(Vec3(-1.0f0, 0.49f0, -110.0f0), Vec3(-1.0f0, 0.49f0, 110.0f0), Vec3(1.0f0, 0.49f0, -110.0f0),
             color = Vec3(0.2f0, 0.0f0, 0.1f0)),
    Triangle(Vec3(1.0f0, 0.49f0, 110.0f0), Vec3(-1.0f0, 0.49f0, 110.0f0), Vec3(1.0f0, 0.49f0, -110.0f0),
             color = Vec3(0.2f0, 0.0f0, 0.1f0)),
    ];

# Render the Environment. This is fed as input to the neural network model
# pos - Tuple(Vector), eg. ([0.0f0], [0.0f0])
# angle - Vector, eg. 
# l - Length of Car, eg. 1.0f0
function renderEnv(pos, angle, l)
    pos_x, pos_z = pos
    lookfrom = Vec3(pos_x, [0.0f0], pos_z)

    lookat_x = (@. Float32(sin(deg2rad(angle))) + pos_x)
    lookat_z = (@. Float32(cos(deg2rad(angle))) + pos_z)
    lookat1 = Vec3(lookat_x, [0.0f0], lookat_z)

    cam = Camera(lookfrom, lookat, Vec3([0.0f0], [1.0f0], [0.0f0]), 10.0f0, 1.0f0, screen_size...)

    origin, direction = get_primary_rays(cam)

    im = raytrace(origin, direction, scene, light, origin, 2)

    color_r = improcess(im.x, screen_size...)
    color_g = improcess(im.y, screen_size...)
    color_b = improcess(im.z, screen_size...)

    im_arr = zeroonenorm(reshape(hcat(color_r, color_g, color_b), screen_size..., 3, 1))

    return im_arr
end

# Rendering an environment
# colorview(RGB, permutedims(renderEnv(([0.0f0], [90.0f0]), [0.0f0], 0.2f0)[:, :, :, 1], (3, 2, 1)))

# Updates the position and angle of the car using the new set of velocities
function _update_pos(posi, angle_arr, wheel_dist, wheelVels, deltaTime)
    angle = angle_arr[1]
    pos = (posi[1][1], posi[2][1])

    Vl, Vr = wheelVels[:, 1]
    l = wheel_dist

    # If the wheel velocities are the same, then there is no rotation
    if Vl == Vr
        pos = pos .+ (deltaTime * Vl) .* (Float32(sin(deg2rad(angle))), Float32(cos(deg2rad(angle))))
        return ([pos[1]], [pos[2]]), [angle]
    end

    # Compute the angular rotation velocity about the ICC (center of curvature)
    w = (Vr - Vl) / l

    # Compute the distance to the center of curvature
    r = (l * (Vl + Vr)) / (2(Vr - Vl))

    # Compute the rotation angle for this time step
    rotAngle = w * deltaTime

    # Rotate the robot's position around the center of rotation
    r_vec_x = Float32(cos(deg2rad(angle)))
    r_vec_z = -Float32(sin(deg2rad(angle)))
    px, pz = pos[1], pos[2]
    cx = px + r * r_vec_x
    cz = pz + r * r_vec_z
    dx = px - cx
    dz = pz - cz

    new_dx = dx * Float32(cos(deg2rad(rotAngle))) + dz * Float32(sin(deg2rad(rotAngle)))
    new_dz = dz * Float32(cos(deg2rad(rotAngle))) - dx * Float32(sin(deg2rad(rotAngle)))
    pos = ([cx + new_dx], [cz + new_dz])

    # Update the robot's direction angle
    angle = (angle + rotAngle)
    return pos, [angle % 360]
end

# Reward Calculation
function reward(pos, pangle_arr, angle_arr, len, pvel, vel)
    angle = angle_arr[1]
    pangle = pangle_arr[1]
    pos_x, pos_z = pos[1][1], pos[2][1]
    rwd = 0.0f0

    mean_vel = mean(vel)
    mean_pvel = mean(pvel)

    # Acceleration Penalty
    rwd -= abs2(mean_vel * Float32(sin(deg2rad(angle))) - mean_pvel * Float32(sin(deg2rad(pangle))))

    # Proximality Reward
    rwd -= abs2(pos_x)

    # Direction Penalty
    rwd -= abs(mean_vel * Float32(sin(deg2rad(angle))))
    rwd += mean_vel * Float32(cos(deg2rad(angle)))

    return rwd
end

function reward(pos, len)
    pos_x, pos_z = pos[1][1], pos[2][1]
    rwd = 0.0f0

    # Penalize for hitting
    if pos_x + len / 2.0f0 > 1.0f0
        rwd -= 10.0f0 * (pos_x + len / 2.0f0)
    end
    if pos_x - len / 2.0f0 < -1.0f0
        rwd += 10.0f0 * (pos_x - len / 2.0f0)
    end

    return rwd
end

# State Struct
mutable struct State
    velo
    angle
    pos
    complete
end

function update_state!(s::State, v, a, p, b::Bool = false)
    s.velo = v
    s.angle = a
    s.pos = p
    s.complete = b
end

# Don't try to differentiate update_state!
Zygote.@nograd update_state!

function episode(model, s::State, l, frames, tstep)
    initial_velo = s.velo
    initial_angle = s.angle
    initial_pos = s.pos
    prev_velo = initial_velo
    new_velo = initial_velo
    new_angle = initial_angle
    prev_angle = initial_angle
    new_pos = initial_pos
    prev_pos = initial_pos
    len = l

    loss = 0

    for frames in 1:frames
        new_pos, new_angle = _update_pos(prev_pos, prev_angle, len, prev_velo, tstep)
        # If leaving road push it back in
        loss = loss - reward(new_pos, len)
        new_pos = (clamp.(new_pos[1], -1.0f0 + l / 2.0f0, 1.0f0 - l / 2.0f0),
                   clamp.(new_pos[2], -100.0f0, Inf32))

        if new_pos[2][1] >= 100.0f0 
            update_state!(s, [0.0f0, 0.0f0], [0.0f0], ([0.0f0], [0.0f0]), true)
            return loss
        end

        env = renderEnv(new_pos, new_angle, len)
        new_velo = model(env) 

        @show new_velo, new_pos, new_angle
        loss = loss - reward(new_pos, prev_angle, new_angle, l, prev_velo, new_velo)
        prev_pos = new_pos
        prev_angle = new_angle
        prev_velo = new_velo
    end

    @show loss

    update_state!(s, new_velo, new_angle, new_pos, false)
    return loss
end

model = Chain(
    Conv((3, 3), 3=>8, relu, pad = 1),
    MaxPool((2, 2)),
    Conv((3, 3), 8=>16, relu, pad = 1),
    MaxPool((2, 2)),
    Conv((3, 3), 16=>32, relu, pad = 1),
    x -> reshape(x, :, 1),
    Dense((32 * 32 * 32), 64, relu),
    Dense(64, 16, relu),
    Dense(16, 2),
    x -> reshape(x, 2))

opt = ADAM(0.001)

for ep in 1:10
    @info "Episode $ep"
    try
        state = State([0.0f0, 0.0f0], [0.0f0], ([0.0f0], [0.0f0]), false)
        dur = 0.01f0
        len = 0.5f0
        for iter in 1:100
            gs = Zygote.gradient(params(model)) do
                episode(model, state, len, 5, dur)
            end
            for arr in params(model)
                Flux.Optimise.update!(opt, arr, gs[arr])
            end
            if iter % 10 == 0
                @info "Subepisode $iter"
            end
            state.complete && break
        end
        @info "Episode Completed"
    catch x
        if isa(x, InterruptException)
            throw(x)
        end
        @info x
        continue
    end
end
