using RayTracer, Zygote, Flux, Images

# Convenience Functions

function create_and_save(color, val)
    img = get_image(color, screen_size.w, screen_size.h)
    save("images/triangle_$(val).jpg", img)
end

# Define the Original Scene Parameters

screen_size = (w = 128, h = 128)

light = PointLight(Vec3(1.0f0), 20000.0f0, Vec3(5.0f0, 5.0f0, -10.0f0))

cam = Camera(Vec3(0.0f0, 0.0f0, -5.0f0), Vec3(0.0f0), Vec3(0.0f0, 1.0f0, 0.0f0),
             45.0f0, 1.0f0, screen_size.w, screen_size.h)

scene = [
    Triangle(Vec3(-1.7f0, 1.0f0, 0.0f0), Vec3(1.0f0, 1.0f0, 0.0f0), Vec3(1.0f0, -1.0f0, 0.0f0),
             color = rgb(1.0f0, 1.0f0, 1.0f0), reflection = 0.5f0)
    ]

origin, direction = get_primary_rays(cam);

color = raytrace(origin, direction, scene, light, origin, 0)

create_and_save(color, "original")

# Perturb the triangle coordinates

scene_new = [
    Triangle(Vec3(-1.9f0, 0.8f0, 0.3f0), Vec3(1.2f0, 0.7f0, 0.4f0), Vec3(1.2f0, -0.5f0, 0.3f0),
             color = rgb(1.0f0, 1.0f0, 1.0f0), reflection = 0.5f0)
    ]

# Approach 1:
# Apply Gaussian Blur on the rendered images before computing loss


function generate_gaussian_filter(σ::T, kernel_size) where {T}
    @assert kernel_size % 2 == 1
    
    gfilter = zeros(T, (kernel_size, kernel_size))
    norm_factor = T(0)
    
    start = - (kernel_size ÷ 2)
    fac = - start + 1
    lst = - start
    
    for x in start:lst,
        y in start:lst
        gfilter[x + fac, y + fac] = exp(-(x^2 + y^2)^2 / (2 * σ^2))
        norm_factor += gfilter[x + fac, y + fac]
    end

    return reshape(gfilter ./ norm_factor, (kernel_size, kernel_size, 1, 1))
end

function process_image(im::Vec3{T}, width, height) where {T}
    im_arr = reshape(hcat(im.x, im.y, im.z), (width, height, 3))
    
    return RayTracer.zeroonenorm(reshape(im_arr, (width, height, 3, 1)))
end

GaussianFilter = generate_gaussian_filter(1.5f0, 3)

GaussianBlur = DepthwiseConv(repeat(GaussianFilter, inner=(1, 1, 1, 3)), [0.0f0], pad = 2)

#blurred_original = GaussianBlur(process_image(color, screen_size.w, screen_size.h))
blurred_original = process_image(color, screen_size.w, screen_size.h)

function loss_fn_gaussian_blur(θ)
    rendered_color = raytrace(origin, direction, θ, light, origin, 0)
    rendered_img_blurred = process_image(rendered_color, screen_size.w, screen_size.h)
    loss = sum(abs2.(rendered_img_blurred .- blurred_original))
    @show loss
    return loss
end

# Generate the initial guess

color_initial_guess = raytrace(origin, direction, scene_new, light, origin, 0)

create_and_save(color_initial_guess, "initial_guess")

# Define the Optimizer and the Optimization Loop

opt = ADAM(0.001f0)

@info "Starting Optimization"

for iter in 1:10000
    global scene_new
    gs = gradient(loss_fn_gaussian_blur, scene_new)[1]
    [update!(opt, scene_new[i], gs[i]) for i in 1:length(scene_new)]
    if iter % 10 == 0
        @info "$(iter) iterations completed."
        create_and_save(raytrace(origin, direction, scene_new, light, origin, 0), iter)
        display(scene_new)
    end
end

@info "Optimization Completed"

