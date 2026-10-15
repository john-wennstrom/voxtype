use super::app::{draw_transcript, SharedState};
use anyhow::Context as _;
use std::sync::Arc;
use std::time::{Duration, Instant};
use voxtype::osd::transcript::{
    now_ms, request_review, ReviewAction, ReviewRequest, TranscriptSnapshot,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
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
        let changed = self.snapshot != snapshot;
        if changed
            && self
                .renderer
                .as_ref()
                .is_some_and(|renderer| renderer.delivery_stage)
        {
            self.renderer = None;
        }
        self.snapshot = snapshot;
        let Some(snapshot) = &self.snapshot else {
            self.renderer = None;
            return;
        };
        if self.renderer.is_none() {
            match Renderer::new(event_loop, snapshot, false) {
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
                    match renderer.render(snapshot) {
                        Ok(Some(PopupAction::Edit)) => {
                            self.renderer = None;
                            match Renderer::new(event_loop, snapshot, false) {
                                Ok(renderer) => self.renderer = Some(renderer),
                                Err(error) => {
                                    self.error = Some(error);
                                    event_loop.exit();
                                }
                            }
                        }
                        Ok(Some(PopupAction::Review(action)))
                            if action.action == ReviewAction::Deliver
                                && !renderer.delivery_stage =>
                        {
                            self.renderer = None;
                            match Renderer::new(event_loop, snapshot, true) {
                                Ok(renderer) => self.renderer = Some(renderer),
                                Err(error) => {
                                    self.error = Some(error);
                                    event_loop.exit();
                                }
                            }
                        }
                        Ok(Some(PopupAction::Review(action))) => {
                            tracing::debug!(?action, "Transcript review button activated");
                            if let Some(path) = &self.shared.transcript_path {
                                if let Err(error) = request_review(path, &action) {
                                    tracing::warn!("Transcript review action failed: {error}");
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            self.error = Some(error);
                            event_loop.exit();
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                renderer.pointer_pos = egui::pos2(position.x as f32, position.y as f32);
                tracing::debug!(position = ?renderer.pointer_pos, "Transcript pointer moved");
                renderer
                    .events
                    .push(egui::Event::PointerMoved(renderer.pointer_pos));
                renderer.window.request_redraw();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                renderer.modifiers = egui::Modifiers {
                    alt: state.alt_key(),
                    ctrl: state.control_key(),
                    shift: state.shift_key(),
                    mac_cmd: false,
                    command: state.control_key(),
                };
            }
            WindowEvent::Focused(focused) => {
                renderer.focused = focused;
                renderer.events.push(egui::Event::WindowFocused(focused));
                renderer.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let Some(key) = egui_key(&event.logical_key) {
                    renderer.events.push(egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed,
                        repeat: event.repeat,
                        modifiers: renderer.modifiers,
                    });
                }
                if pressed && !renderer.modifiers.ctrl && !renderer.modifiers.alt {
                    if let Some(text) = event
                        .text
                        .filter(|text| !text.chars().any(char::is_control))
                    {
                        renderer.events.push(egui::Event::Text(text.to_string()));
                    }
                }
                renderer.window.request_redraw();
            }
            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                renderer.events.push(egui::Event::Text(text));
                renderer.window.request_redraw();
            }
            WindowEvent::CursorLeft { .. } => {
                renderer.events.push(egui::Event::PointerGone);
                renderer.window.request_redraw();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed
                    && !renderer.delivery_stage
                    && self.snapshot.as_ref().is_some_and(|snapshot| {
                        snapshot.settings.review_mode && snapshot.pending_delivery
                    })
                {
                    renderer.window.focus_window();
                }
                tracing::debug!(?state, position = ?renderer.pointer_pos, "Transcript pointer button");
                renderer.events.push(egui::Event::PointerButton {
                    pos: renderer.pointer_pos,
                    button: egui::PointerButton::Primary,
                    pressed: state == ElementState::Pressed,
                    modifiers: Default::default(),
                });
                renderer.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(horizontal, vertical) => {
                        egui::vec2(horizontal, vertical) * 24.0
                    }
                    MouseScrollDelta::PixelDelta(position) => {
                        egui::vec2(position.x as f32, position.y as f32)
                    }
                };
                renderer.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta,
                    phase: egui::TouchPhase::Move,
                    modifiers: Default::default(),
                });
                renderer.window.request_redraw();
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
    events: Vec<egui::Event>,
    pointer_pos: egui::Pos2,
    modifiers: egui::Modifiers,
    focused: bool,
    delivery_stage: bool,
}

enum PopupAction {
    Review(ReviewRequest),
    Edit,
}

fn draw_delivery_confirmation(
    ui: &mut egui::Ui,
    width: u32,
    height: u32,
    snapshot: &TranscriptSnapshot,
) -> Option<PopupAction> {
    ui.style_mut().visuals = egui::Visuals::light();
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width as f32, height as f32));
    ui.painter().rect_filled(rect, 0.0, egui::Color32::WHITE);
    let inner = rect.shrink(24.0_f32.min(height as f32 / 10.0));
    let footer = egui::Rect::from_min_max(egui::pos2(inner.min.x, inner.max.y - 32.0), inner.max);
    let content = egui::Rect::from_min_max(inner.min, egui::pos2(inner.max.x, footer.min.y - 8.0));
    ui.scope_builder(
        egui::UiBuilder::new()
            .id_salt("delivery-preview")
            .max_rect(content),
        |ui| {
            ui.set_clip_rect(content);
            egui::ScrollArea::vertical()
                .max_height(content.height())
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&snapshot.text).size(snapshot.settings.font_size),
                        )
                        .wrap(),
                    );
                });
        },
    );
    let mut action = None;
    ui.scope_builder(
        egui::UiBuilder::new()
            .id_salt("delivery-confirmation")
            .max_rect(footer),
        |ui| {
            ui.set_clip_rect(footer);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        snapshot.pending_delivery,
                        egui::Button::new("Send").min_size(egui::vec2(96.0, 28.0)),
                    )
                    .clicked()
                {
                    action = Some(PopupAction::Review(ReviewRequest {
                        revision: snapshot.revision,
                        action: ReviewAction::Deliver,
                        correction: None,
                    }));
                }
                if ui
                    .add_sized([96.0, 28.0], egui::Button::new("Edit"))
                    .clicked()
                {
                    action = Some(PopupAction::Edit);
                }
            });
        },
    );
    action
}

impl Renderer {
    fn new(
        event_loop: &ActiveEventLoop,
        snapshot: &TranscriptSnapshot,
        delivery_stage: bool,
    ) -> anyhow::Result<Self> {
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
            .with_override_redirect(!snapshot.settings.review_mode || delivery_stage)
            .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
            .with_position(winit::dpi::PhysicalPosition::new(
                origin.x + ((screen.width - width) / 2) as i32,
                origin.y + ((screen.height - height) / 2) as i32,
            ));
        let window = Arc::new(event_loop.create_window(attributes)?);
        window
            .set_cursor_hittest(snapshot.settings.review_mode)
            .context("Make transcript popup click-through")?;
        window.set_ime_allowed(snapshot.settings.review_mode && !delivery_stage);
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
            events: Vec::new(),
            pointer_pos: egui::Pos2::ZERO,
            modifiers: Default::default(),
            focused: false,
            delivery_stage,
        })
    }

    fn render(&mut self, snapshot: &TranscriptSnapshot) -> anyhow::Result<Option<PopupAction>> {
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                self.window.request_redraw();
                return Ok(None);
            }
            other => anyhow::bail!("Popup surface acquisition failed: {other:?}"),
        };
        let width = self.config.width;
        let height = self.config.height;
        let mut action = None;
        let output = self.context.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width as f32, height as f32),
                )),
                events: std::mem::take(&mut self.events),
                modifiers: self.modifiers,
                focused: self.focused,
                ..Default::default()
            },
            |ui| {
                action = if self.delivery_stage {
                    draw_delivery_confirmation(ui, width, height, snapshot)
                } else {
                    draw_transcript(ui, width, height, snapshot).map(PopupAction::Review)
                };
            },
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
        Ok(action)
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

fn egui_key(key: &winit::keyboard::Key) -> Option<egui::Key> {
    use winit::keyboard::{Key, NamedKey};
    Some(match key {
        Key::Named(NamedKey::ArrowLeft) => egui::Key::ArrowLeft,
        Key::Named(NamedKey::ArrowRight) => egui::Key::ArrowRight,
        Key::Named(NamedKey::ArrowUp) => egui::Key::ArrowUp,
        Key::Named(NamedKey::ArrowDown) => egui::Key::ArrowDown,
        Key::Named(NamedKey::Home) => egui::Key::Home,
        Key::Named(NamedKey::End) => egui::Key::End,
        Key::Named(NamedKey::Backspace) => egui::Key::Backspace,
        Key::Named(NamedKey::Delete) => egui::Key::Delete,
        Key::Named(NamedKey::Enter) => egui::Key::Enter,
        Key::Named(NamedKey::Tab) => egui::Key::Tab,
        Key::Named(NamedKey::Escape) => egui::Key::Escape,
        Key::Character(text) if text.eq_ignore_ascii_case("a") => egui::Key::A,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirmation_sends_only_after_an_explicit_click() {
        let snapshot = TranscriptSnapshot {
            text: "Reviewed text".into(),
            revision: 3,
            pending_delivery: true,
            confidence: None,
            words: Vec::new(),
            recording: false,
            expires_at_ms: u64::MAX,
            settings: voxtype::osd::transcript::TranscriptPopupConfig {
                enabled: true,
                review_mode: true,
                ..Default::default()
            },
        };
        for (horizontal, send) in [(72.0, true), (176.0, false)] {
            let context = egui::Context::default();
            let position = egui::pos2(horizontal, 240.0);
            let mut action = None;
            for events in [
                vec![],
                vec![
                    egui::Event::PointerMoved(position),
                    egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: Default::default(),
                    },
                ],
                vec![egui::Event::PointerButton {
                    pos: position,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                }],
            ] {
                let _ = context.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(760.0, 280.0),
                        )),
                        events,
                        ..Default::default()
                    },
                    |ui| {
                        action = draw_delivery_confirmation(ui, 760, 280, &snapshot);
                    },
                );
            }
            match action.unwrap() {
                PopupAction::Review(request) => {
                    assert!(send);
                    assert_eq!(request.action, ReviewAction::Deliver);
                    assert_eq!(request.revision, 3);
                }
                PopupAction::Edit => assert!(!send),
            }
        }
    }

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
