

```
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

// Minimal CubeCL kernel that demonstrates array usage
#[cube(launch)]
fn array_multiply_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = ABSOLUTE_POS;

    if tid < input.len() && tid < output.len() {
        // Simple computation that demonstrates array access
        // Each element is multiplied by its position + 2
        let multiplier = (tid + 2) as f32;
        output[tid] = input[tid] * multiplier;
    }
}

fn main() {
    // Initialize CubeCL environment with WGPU backend
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    // Create a fixed-size input array - this demonstrates array declaration and usage
    let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    // Convert array to bytes for buffer creation
    let input_bytes = bytemuck::cast_slice(&input_data);
    let output_size = input_data.len() * std::mem::size_of::<f32>();

    // Create input and output buffers
    let input_buffer = client.create(input_bytes);
    let output_buffer = client.empty(output_size);

    // Launch the kernel
    array_multiply_kernel::launch::<WgpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(4, 1, 1),
        unsafe { ArrayArg::from_raw_parts::<f32>(&input_buffer, input_data.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&output_buffer, input_data.len(), 1) },
    );

    // Read results back from GPU
    let output_bytes = client.read(vec![output_buffer.binding()]);
    let output_data: &[f32] = bytemuck::cast_slice(&output_bytes[0]);

    println!("Input array:  {:?}", input_data);
    println!("Output array: {:?}", output_data);
    println!("Array multiplication completed successfully!");
}

```