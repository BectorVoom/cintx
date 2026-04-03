#type/docs
#Rust/cubecl/multi
```rust
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

#[cube(launch_unchecked)]
fn vector_add(lhs: &Array<f32>, rhs: &Array<f32>, output: &mut Array<f32>) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        output[index] = lhs[index] + rhs[index];
    }
}

#[tokio::main]
async fn main() {
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let size = 1024;
    let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    let lhs = client.create(bytemuck::cast_slice(&lhs_data));
    let rhs = client.create(bytemuck::cast_slice(&rhs_data));
    let output = client.empty(size * std::mem::size_of::<f32>());
    let wg = 256usize;
    let groups = (size as usize + wg - 1) / wg;
    unsafe {
        vector_add::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(groups as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&lhs, groups, 1),
            ArrayArg::from_raw_parts::<f32>(&rhs, groups, 1),
            ArrayArg::from_raw_parts::<f32>(&output, groups, 1),
        );
    }

    client.sync().await;
}

```