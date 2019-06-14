using RayTracer, Images, Flux, Zygote, Statistics
using RayTracer: improcess

screen_size = (w = 512, h = 512)

light = DistantLight(Vec3(1.0f0), 100.0f0, Vec3(0.0f0, 1.0f0, 0.0f0))

scene = [
    SimpleSphere(Vec3(0.0f0, -99999.5f0, 0.0f0), 99999.0f0, color = Vec3(0.0f0, 1.0f0, 0.0f0)),
    Triangle(Vec3(-1.0f0, 0.49f0, -40.0f0), Vec3(-1.0f0, 0.49f0, 40.0f0), Vec3(1.0f0, 0.49f0, -40.0f0),
             color = Vec3(0.2f0, 0.0f0, 0.1f0)),
    Triangle(Vec3(1.0f0, 0.49f0, 40.0f0), Vec3(-1.0f0, 0.49f0, 40.0f0), Vec3(1.0f0, 0.49f0, -40.0f0),
             color = Vec3(0.2f0, 0.0f0, 0.1f0))
   ]

# pos -> (pos_x, pos_z)
# angle -> [angle wrt forward direction]
function renderEnv(pos, angle, l)
    pos_x, pos_z = pos
    lookfrom1 = Vec3(pos_x, [0.0f0], pos_z)

    lookat_x1 = (@. Float32(sin(deg2rad(angle))) + pos_x)
    lookat_z1 = (@. Float32(cos(deg2rad(angle))) + pos_z)
    lookat1 = Vec3(lookat_x1, [0.0f0], lookat_z1)

    cam1 = Camera(lookfrom1, lookat1, Vec3([0.0f0], [1.0f0], [0.0f0]), 10.0f0, 1.0f0, screen_size...)

    origin1, direction1 = get_primary_rays(cam1)

    im1 = raytrace(origin1, direction1, scene, light, origin1, 2)

    color_r = improcess(im1.x, screen_size...)
    color_g = improcess(im1.y, screen_size...)
    color_b = improcess(im1.z, screen_size...)

    im_arr1 = zeroonenorm(reshape(hcat(color_r, color_g, color_b), screen_size..., 3, 1))

    return im_arr1
end

save("tt.jpg", colorview(RGB, permutedims(renderEnv(([0.0f0], [0.0f0]), [0.0f0], 0.2f0)[:, :, :, 1], (3, 2, 1))))

