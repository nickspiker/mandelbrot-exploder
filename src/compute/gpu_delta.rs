//! wgpu/Metal runner for the scaled f32 delta loop. The kernel (`delta.wgsl`) is `run_f32`;
//! this file only moves `DeltaState` arrays and the reference orbit across and back.
//!
//! Each `run` is synchronous: upload states, dispatch, copy to a readback buffer, map, copy
//! out. On unified memory that round trip is a few milliseconds for 10⁵ pixels, well under
//! the kernel time for a 2048-iteration chunk.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::perturbation::{DeltaRunner, DeltaState, Reference};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    s: f32,
    ref_last: u32,
    periodic: u32,
    count: u32,
}

pub struct GpuRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    adapter_name: String,
    /// (identity of the uploaded reference, buffer, entry count)
    zref: Option<(usize, wgpu::Buffer, u32)>,
    /// (state buffer, readback buffer, capacity in states)
    states: Option<(wgpu::Buffer, wgpu::Buffer, usize)>,
}

impl GpuRunner {
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("delta"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("delta.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("delta.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("delta layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("delta pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("delta"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("delta params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            device,
            queue,
            pipeline,
            layout,
            params,
            adapter_name,
            zref: None,
            states: None,
        })
    }

    fn ensure_reference(&mut self, r: &Reference) {
        let id = r.z32.as_ptr() as usize ^ (r.z32.len() << 1);
        if self.zref.as_ref().map(|(i, _, _)| *i) == Some(id) {
            return;
        }
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("delta reference"),
            contents: bytemuck::cast_slice(&r.z32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.zref = Some((id, buf, r.z32.len() as u32));
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.states.as_ref().map(|(_, _, c)| *c >= n).unwrap_or(false) {
            return;
        }
        let cap = n.next_power_of_two().max(1 << 12);
        let bytes = (cap * size_of::<DeltaState>()) as u64;
        let states = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("delta states"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("delta readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.states = Some((states, readback, cap));
    }
}

impl DeltaRunner for GpuRunner {
    fn name(&self) -> String {
        format!("GPU ({})", self.adapter_name)
    }

    fn run(&mut self, r: &Reference, s: f32, states: &mut [DeltaState]) {
        if states.is_empty() {
            return;
        }
        self.ensure_reference(r);
        self.ensure_capacity(states.len());
        let (_, zref, len) = self.zref.as_ref().unwrap();
        let (state_buf, readback, _) = self.states.as_ref().unwrap();
        let bytes = (states.len() * size_of::<DeltaState>()) as u64;

        self.queue.write_buffer(state_buf, 0, bytemuck::cast_slice(states));
        let params = Params {
            s,
            ref_last: len - 1,
            periodic: r.period.is_some() as u32,
            count: states.len() as u32,
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("delta bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: zref.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: state_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(bytes),
                    }),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("delta") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("delta"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(states.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(state_buf, 0, readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        {
            let data = slice.get_mapped_range();
            states.copy_from_slice(bytemuck::cast_slice(&data));
        }
        readback.unmap();
    }
}
