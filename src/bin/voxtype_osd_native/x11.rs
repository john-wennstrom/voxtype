use super::app::{draw_transcript, SharedState};
use anyhow::Context as _;
use std::sync::Arc;
use std::time::{Duration, Instant};
use voxtype::osd::transcript::{now_ms, TranscriptSnapshot};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::x11::{EventLoopBuilderExtX11, WindowAttributesExtX11};
use winit::window::{Window, WindowId, WindowLevel};

pub fn run(shared: SharedState) -> anyhow::Result<()> {
    let event_loop = EventLoop::builder()
        .with_x11()
        .build()
        .context("connect to XWayland; is DISPLAY set?")?;
    let mut application = Popup {
        shared,
        renderer: None,
        snapshot: None,
        error: None,
    };
    event_loop.run_app(&mut application)?;
    match application.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

struct Popup {
    shared: SharedState,
    renderer: Option<Renderer>,
    snapshot: Option<TranscriptSnapshot>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for Popup {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(50),
        ));
        let snapshot = self
            .shared
            .transcript_path
            .as_deref()
            .and_then(TranscriptSnapshot::read)
            .filter(|value| value.visible_at(now_ms()));
        let changed = self.snapshot.as_ref().map(|value| &value.text)
            != snapshot.as_ref().map(|value| &value.text);
        self.snapshot = snapshot;
        let Some(snapshot) = &self.snapshot else {
            self.renderer = None;
            return;
        };
        if self.renderer.is_none() {
            match Renderer::new(event_loop, snapshot) {
                Ok(renderer) => self.renderer = Some(renderer),
                Err(error) => {
                    self.error = Some(error);
                    event_loop.exit();
                    return;
                }
            }
        }
        if changed {
            self.renderer.as_ref().unwrap().window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self
            .renderer
            .as_mut()
            .filter(|renderer| renderer.window.id() == window_id)
        else {
            return;
        };
        match event {
            WindowEvent::RedrawRequested => {
                if let Some(snapshot) = &self.snapshot {
                    if let Err(error) = renderer.render(snapshot) {
                        self.error = Some(error);
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::Resized(size) => {
                renderer.config.width = size.width.max(1);
                renderer.config.height = size.height.max(1);
                renderer
                    .surface
                    .configure(&renderer.device, &renderer.config);
                renderer.window.request_redraw();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}

struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    context: egui::Context,
    painter: egui_wgpu::Renderer,
    window: Arc<Window>,
}

impl Renderer {
    fn new(event_loop: &ActiveEventLoop, snapshot: &TranscriptSnapshot) -> anyhow::Result<Self> {
        let monitor = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
            .context("No display output for transcript popup")?;
        let screen = monitor.size();
        let width = snapshot
            .settings
            .width_px
            .min(screen.width.saturating_sub(48).max(1));
        let height = snapshot
            .settings
            .height_px
            .min(screen.height.saturating_sub(48).max(1));
        let origin = monitor.position();
        let attributes = Window::default_attributes()
            .with_title("Voxtype Transcript")
            .with_decorations(false)
            .with_transparent(false)
            .with_active(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_override_redirect(true)
            .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
            .with_position(winit::dpi::PhysicalPosition::new(
                origin.x + ((screen.width - width) / 2) as i32,
                origin.y + ((screen.height - height) / 2) as i32,
            ));
        let window = Arc::new(event_loop.create_window(attributes)?);
        window
            .set_cursor_hittest(false)
            .context("Make transcript popup click-through")?;
        window.set_ime_allowed(false);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            }))?;
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .context("No supported popup surface format")?;
        config.format =
            popup_surface_format(&surface.get_capabilities(&adapter).formats, config.format);
        config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        surface.configure(&device, &config);
        let painter = egui_wgpu::Renderer::new(
            &device,
            config.format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );
        window.request_redraw();
        Ok(Self {
            surface,
            device,
            queue,
            config,
            context: egui::Context::default(),
            painter,
            window,
        })
    }

    fn render(&mut self, snapshot: &TranscriptSnapshot) -> anyhow::Result<()> {
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                self.window.request_redraw();
                return Ok(());
            }
            other => anyhow::bail!("Popup surface acquisition failed: {other:?}"),
        };
        let width = self.config.width;
        let height = self.config.height;
        let output = self.context.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width as f32, height as f32),
                )),
                ..Default::default()
            },
            |ui| draw_transcript(ui, width, height, snapshot),
        );
        let primitives = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: output.pixels_per_point,
        };
        for (texture_id, delta) in &output.textures_delta.set {
            self.painter
                .update_texture(&self.device, &self.queue, *texture_id, delta);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let command_buffers = self.painter.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &primitives,
            &screen,
        );
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.painter.render(&mut pass, &primitives, &screen);
        }
        self.queue.submit(
            command_buffers
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        for texture_id in output.textures_delta.free {
            self.painter.free_texture(&texture_id);
        }
        texture.present();
        Ok(())
    }
}

fn popup_surface_format(
    formats: &[wgpu::TextureFormat],
    fallback: wgpu::TextureFormat,
) -> wgpu::TextureFormat {
    formats
        .iter()
        .copied()
        .find(|format| {
            matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
            )
        })
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_popup_prefers_egui_surface_format() {
        use wgpu::TextureFormat::{Bgra8Unorm, Bgra8UnormSrgb, Rgba8Unorm};
        assert_eq!(
            popup_surface_format(&[Bgra8UnormSrgb, Bgra8Unorm], Bgra8UnormSrgb),
            Bgra8Unorm
        );
        assert_eq!(
            popup_surface_format(&[Rgba8Unorm], Bgra8UnormSrgb),
            Rgba8Unorm
        );
        assert_eq!(
            popup_surface_format(&[Bgra8UnormSrgb], Bgra8UnormSrgb),
            Bgra8UnormSrgb
        );
    }
}
