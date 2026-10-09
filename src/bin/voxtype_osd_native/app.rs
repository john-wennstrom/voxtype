//! Wayland + wgpu + egui-wgpu glue for `voxtype-osd-native`.
//!
//! The whole rendering stack is collapsed into one file because the borrow
//! relationships between SCTK state, the wgpu device/queue, the surface
//! configuration, and the egui-wgpu renderer are awkward to split without
//! introducing references with non-trivial lifetimes. Each piece is small,
//! and keeping them together makes the lifecycle (`create_surface_if_needed`
//! / `tear_down_surface`) easy to follow.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context as _};
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat,
    output::{OutputHandler, OutputState},
    reexports::{
        calloop::{self, EventLoop},
        calloop_wayland_source::WaylandSource,
        client::{
            globals::registry_queue_init,
            protocol::{wl_output, wl_pointer, wl_seat, wl_surface::WlSurface},
            Connection, Proxy, QueueHandle,
        },
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        pointer::{PointerEvent, PointerEventKind, PointerHandler},
        Capability, SeatHandler, SeatState,
    },
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
};

use voxtype::audio::levels::AudioFrame;
use voxtype::osd::config::{OsdConfig, OsdPosition};
use voxtype::osd::ipc::FrameRing;
use voxtype::osd::transcript::{now_ms, request_review_action, ReviewAction, TranscriptSnapshot};
use voxtype::osd::visual::{
    peak_meter_fraction, project_envelope, EnvelopeColumn, MeterZone, Palette, PeakHold,
};

/// State shared between the IPC thread and the render thread.
#[derive(Clone)]
pub struct SharedState {
    pub ring: Arc<Mutex<FrameRing>>,
    pub peak_hold: Arc<Mutex<PeakHold>>,
    /// Wall-clock timestamp of the most recent frame. Used to drive idle
    /// teardown when no frames have arrived for a while.
    pub last_frame_at: Arc<Mutex<Option<Instant>>>,
    pub palette: Palette,
    pub config: OsdConfig,
    pub transcript_path: Option<std::path::PathBuf>,
    pub transcript: Arc<Mutex<Option<TranscriptSnapshot>>>,
}

/// How long to keep the surface alive after the last frame arrived, before
/// destroying it. The daemon stops emitting between recordings; this value
/// controls how quickly the OSD disappears after that.
/// Idle threshold for tearing down the layer-shell + wgpu surface. Set
/// short enough that the OSD disappears immediately when the user releases
/// the hotkey, but long enough that the destroy+recreate cost on the next
/// recording isn't visible. 0.5s is the sweet spot: humans perceive sub-
/// second as "instant," and 0.5s is well above the recording boundary
/// gaps the daemon naturally produces.
const IDLE_TEARDOWN_SECS: f32 = 0.5;
/// Idle/recovery checks only; animation is paced by Wayland frame callbacks.
const IDLE_CHECK_INTERVAL_MS: u64 = 50;

/// Outer state owned by the calloop event loop. Implements the SCTK delegate
/// traits via `delegate_*` macros.
pub struct App {
    registry_state: RegistryState,
    output_state: OutputState,
    compositor_state: CompositorState,
    layer_shell: LayerShell,
    seat_state: SeatState,
    pointers: Vec<(wl_seat::WlSeat, wl_pointer::WlPointer)>,
    pointer_events: Vec<egui::Event>,

    qh: QueueHandle<App>,
    conn: Connection,

    shared: SharedState,
    surface: Option<RenderSurface>,
    transcript_dirty: bool,
}

/// All state tied to the live layer-shell surface. Dropped (via
/// `Option::take`) when we tear down for idle.
struct RenderSurface {
    layer: LayerSurface,
    wl_surface: WlSurface,

    /// Last accepted size from the compositor's configure. We use this to
    /// configure the wgpu surface.
    width: u32,
    height: u32,
    /// Whether we've received the first configure (and thus may render).
    configured: bool,
    frame_pending: bool,

    // wgpu plumbing.
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,

    // egui plumbing.
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
}

/// Run the event loop. Returns when the user closes the surface or the
/// loop exits via signal.
pub fn run(
    shared: SharedState,
    frame_ping_source: calloop::ping::PingSource,
) -> anyhow::Result<()> {
    let conn =
        Connection::connect_to_env().context("connect to Wayland; is WAYLAND_DISPLAY set?")?;
    let (globals, event_queue) = registry_queue_init::<App>(&conn).context("init registry")?;
    let qh = event_queue.handle();

    let mut event_loop: EventLoop<'static, App> =
        EventLoop::try_new().context("create calloop event loop")?;
    let loop_handle = event_loop.handle();

    let compositor_state =
        CompositorState::bind(&globals, &qh).context("compositor protocol unavailable")?;
    let layer_shell =
        LayerShell::bind(&globals, &qh).context("wlr-layer-shell protocol unavailable")?;
    let output_state = OutputState::new(&globals, &qh);
    let registry_state = RegistryState::new(&globals);

    WaylandSource::new(conn.clone(), event_queue)
        .insert(loop_handle.clone())
        .map_err(|e| anyhow!("insert WaylandSource: {}", e))?;

    let mut app = App {
        registry_state,
        output_state,
        compositor_state,
        layer_shell,
        seat_state: SeatState::new(&globals, &qh),
        pointers: Vec::new(),
        pointer_events: Vec::new(),
        qh: qh.clone(),
        conn: conn.clone(),
        shared,
        surface: None,
        transcript_dirty: false,
    };

    // Wake on each incoming audio frame: create the surface if needed,
    // request a redraw.
    loop_handle
        .insert_source(frame_ping_source, move |_, _, app: &mut App| {
            app.on_frame_ping();
        })
        .map_err(|e| anyhow!("insert ping source: {}", e))?;

    // Periodic idle teardown and recovery after a skipped surface acquisition.
    let timer = calloop::timer::Timer::from_duration(Duration::from_millis(IDLE_CHECK_INTERVAL_MS));
    loop_handle
        .insert_source(timer, |_deadline, _, app: &mut App| {
            app.tick();
            calloop::timer::TimeoutAction::ToDuration(Duration::from_millis(IDLE_CHECK_INTERVAL_MS))
        })
        .map_err(|e| anyhow!("insert redraw timer: {}", e))?;

    tracing::info!("entering event loop");
    loop {
        if let Err(e) = event_loop.dispatch(Some(Duration::from_secs(1)), &mut app) {
            tracing::error!("event loop dispatch failed: {}", e);
            break;
        }
    }

    drop(app);
    drop(conn);
    Ok(())
}

impl App {
    fn on_frame_ping(&mut self) {
        if self.surface.is_none() {
            if let Err(e) = self.create_surface() {
                tracing::warn!("Failed to create OSD surface: {:#}", e);
            }
        }
    }

    fn tick(&mut self) {
        if let Some(path) = &self.shared.transcript_path {
            let snapshot =
                TranscriptSnapshot::read(path).filter(|value| value.visible_at(now_ms()));
            let mut current = self.shared.transcript.lock().expect("transcript poisoned");
            self.transcript_dirty |= current.as_ref() != snapshot.as_ref();
            *current = snapshot;
            let visible = current.is_some();
            if let Some(value) = current.as_ref() {
                self.shared.config.width_px = value.settings.width_px;
                self.shared.config.height_px = value.settings.height_px;
            }
            drop(current);
            if !visible {
                self.tear_down_surface();
            } else {
                if self.surface.is_none() {
                    if let Err(error) = self.create_surface() {
                        tracing::warn!("Failed to create transcript popup: {error:#}");
                        return;
                    }
                }
                if self.transcript_dirty {
                    if let Err(error) = self.render_frame() {
                        tracing::warn!("Transcript render failed: {error:#}");
                    }
                }
            }
            return;
        }
        let last_frame = self.shared.last_frame_at.lock().ok().and_then(|g| *g);
        let idle = match last_frame {
            Some(t) => t.elapsed().as_secs_f32() >= IDLE_TEARDOWN_SECS,
            None => true,
        };

        if idle && self.surface.is_some() {
            tracing::info!("Idle for {}s, tearing down surface", IDLE_TEARDOWN_SECS);
            self.tear_down_surface();
            return;
        }

        if self.surface.is_some() && !idle {
            if let Err(e) = self.render_frame() {
                tracing::warn!("render failed: {:#}", e);
            }
        }
    }

    fn create_surface(&mut self) -> anyhow::Result<()> {
        tracing::info!("Creating OSD layer surface");

        let wl_surface = self.compositor_state.create_surface(&self.qh);
        let layer = self.layer_shell.create_layer_surface(
            &self.qh,
            wl_surface.clone(),
            Layer::Overlay,
            Some(if self.shared.transcript_path.is_some() {
                "voxtype-transcript"
            } else {
                "voxtype-osd"
            }),
            None,
        );

        if self.shared.transcript_path.is_some() {
            if let Some((width, height)) = self
                .output_state
                .outputs()
                .filter_map(|output| {
                    self.output_state
                        .info(&output)
                        .and_then(|info| info.logical_size)
                })
                .min()
            {
                self.shared.config.width_px = self
                    .shared
                    .config
                    .width_px
                    .min((width.max(1) as u32).saturating_sub(48).max(1));
                self.shared.config.height_px = self
                    .shared
                    .config
                    .height_px
                    .min((height.max(1) as u32).saturating_sub(48).max(1));
            }
        }
        let cfg = &self.shared.config;
        let (anchor, margin_top, margin_bottom, margin_left, margin_right) =
            position_to_anchor_and_margins(cfg.position, cfg.margin_px as i32);
        if self.shared.transcript_path.is_some() {
            layer.set_anchor(Anchor::empty());
            layer.set_margin(0, 0, 0, 0);
        } else {
            layer.set_anchor(anchor);
            layer.set_margin(margin_top, margin_right, margin_bottom, margin_left);
        }
        layer.set_size(cfg.width_px, cfg.height_px);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.set_exclusive_zone(0);

        // Empty input region — clicks pass through. SCTK's Region helper
        // owns the wl_region and destroys it on drop. The wl_region must
        // outlive the commit that activates it; we let it drop after.
        let region = Region::new(&self.compositor_state)
            .map_err(|e| anyhow!("create input region: {}", e))?;
        let review_mode = self
            .shared
            .transcript
            .lock()
            .ok()
            .and_then(|snapshot| {
                snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.settings.review_mode)
            })
            .unwrap_or(false);
        if !review_mode {
            wl_surface.set_input_region(Some(region.wl_region()));
        }

        layer.commit();
        drop(region);

        // wgpu instance + surface.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        // Raw handles. With wayland-client's `system` feature + wayland-backend
        // `client_system`, ObjectId/Backend expose libwayland pointers.
        let display_ptr = NonNull::new(self.conn.backend().display_ptr() as *mut std::ffi::c_void)
            .ok_or_else(|| anyhow!("null wl_display ptr"))?;
        let surface_ptr = NonNull::new(wl_surface.id().as_ptr() as *mut std::ffi::c_void)
            .ok_or_else(|| anyhow!("null wl_surface ptr"))?;

        let raw_display = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(display_ptr));
        let raw_window = RawWindowHandle::Wayland(WaylandWindowHandle::new(surface_ptr));

        // SAFETY: the `wl_display` and `wl_surface` outlive the wgpu surface
        // because `RenderSurface` keeps them alive (Connection is held in
        // `App`; wl_surface is held in RenderSurface).
        let surface = unsafe {
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_display),
                raw_window_handle: raw_window,
            })
        }
        .context("create wgpu surface")?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("request wgpu adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("voxtype-osd-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("request wgpu device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(f, wgpu::TextureFormat::Bgra8UnormSrgb))
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| anyhow!("no surface formats available"))?;

        let egui_ctx = egui::Context::default();
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );

        self.surface = Some(RenderSurface {
            layer,
            wl_surface,
            width: cfg.width_px,
            height: cfg.height_px,
            configured: false,
            frame_pending: false,
            _instance: instance,
            surface,
            device,
            queue,
            surface_format,
            egui_ctx,
            egui_renderer,
        });

        Ok(())
    }

    fn tear_down_surface(&mut self) {
        if let Some(rs) = self.surface.take() {
            let RenderSurface {
                layer,
                wl_surface,
                surface,
                device,
                queue,
                egui_renderer,
                _instance,
                ..
            } = rs;
            // Drop wgpu state first, then the wl_surface. LayerSurface drops
            // the role on drop; we then explicitly destroy the wl_surface.
            drop(egui_renderer);
            drop(queue);
            drop(device);
            drop(surface);
            drop(_instance);
            drop(layer);
            wl_surface.destroy();
        }
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let rs = match self.surface.as_mut() {
            Some(s) if s.configured && !s.frame_pending => s,
            _ => return Ok(()),
        };

        let cst = rs.surface.get_current_texture();
        let surface_texture: wgpu::SurfaceTexture = match cst {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                reconfigure_surface(rs);
                return Ok(());
            }
            other => {
                tracing::debug!("acquire frame skipped: {:?}", other);
                return Ok(());
            }
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let raw_input = egui::RawInput {
            events: std::mem::take(&mut self.pointer_events),
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(rs.width as f32, rs.height as f32),
            )),
            ..Default::default()
        };

        let palette = self.shared.palette;
        let cfg = &self.shared.config;
        let waveform_window_secs = cfg.waveform_window_secs;
        let meter_w = ((rs.width as f32) * 0.05).max(8.0);
        let waveform_w = (rs.width as f32) - meter_w - 4.0;
        let n_columns = waveform_w.max(32.0) as usize;

        let envelope_cols = {
            let ring = self.shared.ring.lock().expect("ring poisoned");
            let buf: Vec<AudioFrame> = ring.iter().collect();
            project_envelope(
                &buf,
                n_columns,
                waveform_window_secs as f64 * voxtype::audio::levels::FRAME_HZ as f64,
                ring.scroll_offset(Instant::now()),
            )
        };

        let (peak_dbfs, held_dbfs) = {
            let ring = self.shared.ring.lock().expect("ring poisoned");
            let p = ring.latest().map(|f| f.peak_dbfs).unwrap_or(-120.0);
            let h = self
                .shared
                .peak_hold
                .lock()
                .map(|x| x.held_dbfs)
                .unwrap_or(-120.0);
            (p, h)
        };

        let width_px = rs.width;
        let height_px = rs.height;
        let gain = self.shared.config.waveform_gain;
        let transcript = self
            .shared
            .transcript
            .lock()
            .ok()
            .and_then(|value| value.clone());
        let mut review_action = None;
        let full_output = rs.egui_ctx.run_ui(raw_input, |ui| {
            if let Some(snapshot) = &transcript {
                review_action = draw_transcript(ui, width_px, height_px, snapshot);
            } else {
                draw_ui(
                    ui,
                    width_px,
                    height_px,
                    &palette,
                    &envelope_cols,
                    peak_dbfs,
                    held_dbfs,
                    gain,
                );
            }
        });

        let primitives = rs
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [rs.width, rs.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            rs.egui_renderer
                .update_texture(&rs.device, &rs.queue, *id, image_delta);
        }

        let mut encoder = rs
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxtype-osd-encoder"),
            });

        rs.egui_renderer.update_buffers(
            &rs.device,
            &rs.queue,
            &mut encoder,
            &primitives,
            &screen_descriptor,
        );

        {
            let bg = if transcript.is_some() {
                voxtype::osd::visual::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                }
            } else {
                palette.background
            };
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("voxtype-osd-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: bg.r as f64,
                                g: bg.g as f64,
                                b: bg.b as f64,
                                a: bg.a as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();

            rs.egui_renderer
                .render(&mut rpass, &primitives, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            rs.egui_renderer.free_texture(id);
        }

        rs.queue.submit(Some(encoder.finish()));
        rs.wl_surface.frame(&self.qh, rs.wl_surface.clone());
        rs.frame_pending = true;
        surface_texture.present();
        self.transcript_dirty = false;
        if let (Some(action), Some(snapshot), Some(path)) = (
            review_action,
            transcript.as_ref(),
            self.shared.transcript_path.as_ref(),
        ) {
            if let Err(error) = request_review_action(path, snapshot.revision, action) {
                tracing::warn!("Transcript review action failed: {error}");
            }
        }
        Ok(())
    }
}

pub(crate) fn draw_transcript(
    ui: &mut egui::Ui,
    width: u32,
    height: u32,
    snapshot: &TranscriptSnapshot,
) -> Option<ReviewAction> {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width as f32, height as f32));
    let painter = ui.painter().clone();
    painter.rect_filled(
        rect,
        0.0,
        egui::Color32::from_white_alpha((snapshot.settings.opacity * 255.0).round() as u8),
    );
    let mut inner = rect.shrink(24.0_f32.min(height as f32 / 10.0));
    let footer = egui::Rect::from_min_max(egui::pos2(inner.min.x, inner.max.y - 32.0), inner.max);
    if snapshot.settings.review_mode {
        inner.max.y -= 40.0;
    }
    let text_color = egui::Color32::from_rgb(24, 24, 24);
    let score_color = egui::Color32::from_rgb(68, 86, 92);
    if snapshot.settings.review_mode || snapshot.confidence.is_some() {
        let label = snapshot.confidence.map_or_else(
            || "Confidence --/100".to_string(),
            |score| format!("Confidence {:.0}/100", score * 100.0),
        );
        painter.with_clip_rect(inner).text(
            inner.left_top(),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(14.0),
            score_color,
        );
        inner.min.y += 24.0;
    }
    let normal_format = egui::TextFormat {
        font_id: egui::FontId::proportional(snapshot.settings.font_size),
        color: text_color,
        ..Default::default()
    };
    let score_format = egui::TextFormat {
        font_id: egui::FontId::proportional((snapshot.settings.font_size * 0.55).max(12.0)),
        color: score_color,
        ..Default::default()
    };
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = inner.width().max(1.0);
    if snapshot.words.is_empty() {
        job.append(&snapshot.text, 0.0, normal_format);
    } else {
        for (index, word) in snapshot.words.iter().enumerate() {
            if index > 0 {
                job.append(" ", 0.0, normal_format.clone());
            }
            job.append(&word.text, 0.0, normal_format.clone());
            if let Some(score) = word.confidence {
                job.append(
                    &format!("\u{00a0}[{:.0}]", score * 100.0),
                    0.0,
                    score_format.clone(),
                );
            }
        }
    }
    if snapshot.settings.review_mode {
        ui.style_mut().visuals = egui::Visuals::light();
        ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
            ui.set_clip_rect(inner);
            egui::ScrollArea::vertical()
                .id_salt("review-transcript")
                .max_height(inner.height().max(1.0))
                .auto_shrink([false, false])
                .stick_to_bottom(snapshot.recording)
                .show(ui, |ui| {
                    ui.add(egui::Label::new(job).wrap());
                });
        });
        let mut action = None;
        ui.scope_builder(egui::UiBuilder::new().max_rect(footer), |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add_enabled_ui(snapshot.pending_delivery, |ui| {
                    let deliver = egui::Button::new(
                        egui::RichText::new("Deliver").color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(34, 103, 94));
                    if ui.add_sized([96.0, 28.0], deliver).clicked() {
                        action = Some(ReviewAction::Deliver);
                    }
                    if ui
                        .add_sized([96.0, 28.0], egui::Button::new("Discard"))
                        .clicked()
                    {
                        action = Some(ReviewAction::Discard);
                    }
                });
            });
        });
        return action;
    }
    let galley = painter.layout_job(job);
    let offset = (galley.size().y - inner.height()).max(0.0);
    painter.with_clip_rect(inner).galley(
        inner.left_top() - egui::vec2(0.0, offset),
        galley,
        text_color,
    );
    None
}

#[cfg(test)]
mod transcript_tests {
    use super::*;
    use voxtype::osd::transcript::TranscriptPopupConfig;

    #[test]
    fn transcript_popup_wraps_with_square_translucent_background() {
        for width in [320, 760] {
            let snapshot = TranscriptSnapshot {
                text: "A live transcript wraps without changing the focused window. ".repeat(12),
                revision: 0,
                pending_delivery: false,
                confidence: None,
                words: Vec::new(),
                recording: true,
                expires_at_ms: 0,
                settings: TranscriptPopupConfig {
                    enabled: true,
                    ..Default::default()
                },
            };
            let context = egui::Context::default();
            let output = context.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, 280.0),
                    )),
                    ..Default::default()
                },
                |ui| {
                    draw_transcript(ui, width, 280, &snapshot);
                },
            );
            let background = output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::Shape::Rect(rectangle) => Some(rectangle),
                    _ => None,
                })
                .unwrap();
            assert_eq!(background.corner_radius, egui::CornerRadius::ZERO);
            assert_eq!(background.stroke.width, 0.0);
            assert_eq!(background.fill.a(), 204);
            let text = output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::Shape::Text(text) => Some(text),
                    _ => None,
                })
                .unwrap();
            assert!(text.galley.rows.len() > 1);
            assert!(text.galley.size().x <= width as f32 - 48.0);
        }
    }

    #[test]
    fn transcript_popup_renders_numeric_confidence_without_overlap() {
        for (width, height) in [(240, 80), (320, 280), (760, 280), (1920, 1080)] {
            let snapshot = TranscriptSnapshot {
                text: "Hello world".to_string(),
                revision: 0,
                pending_delivery: false,
                confidence: Some(0.7),
                words: vec![
                    voxtype::transcribe::WordConfidence {
                        text: "Hello".to_string(),
                        confidence: Some(0.8),
                    },
                    voxtype::transcribe::WordConfidence {
                        text: "world".to_string(),
                        confidence: Some(0.6),
                    },
                ],
                recording: true,
                expires_at_ms: 0,
                settings: TranscriptPopupConfig {
                    enabled: true,
                    ..Default::default()
                },
            };
            let context = egui::Context::default();
            let output = context.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, height as f32),
                    )),
                    ..Default::default()
                },
                |ui| {
                    draw_transcript(ui, width, height, &snapshot);
                },
            );
            let texts: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| {
                    if let egui::Shape::Text(text) = &shape.shape {
                        Some((shape, text))
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(texts.len(), 2);
            assert_eq!(texts[0].1.galley.job.text, "Confidence 70/100");
            assert_eq!(
                texts[1].1.galley.job.text,
                "Hello\u{00a0}[80] world\u{00a0}[60]"
            );
            assert!(texts[1]
                .1
                .galley
                .rows
                .iter()
                .all(|row| { row.glyphs.first().is_none_or(|glyph| glyph.chr != '[') }));
            let padding = 24.0_f32.min(height as f32 / 10.0);
            assert!(texts[0].1.galley.size().x <= width as f32 - padding * 2.0);
            assert!(texts[1].1.galley.size().x <= width as f32 - padding * 2.0);
            assert!(texts[0].1.pos.y + texts[0].1.galley.size().y <= texts[1].0.clip_rect.min.y);
            assert!(texts[1].0.clip_rect.max.y <= height as f32 - padding);
        }
    }
    #[test]
    fn transcript_popup_review_always_renders_confidence_header() {
        for (text, recording, pending_delivery) in [
            ("", false, false),
            ("Live words", true, false),
            ("Reviewed words", false, true),
        ] {
            for confidence in [None, Some(0.85)] {
                let snapshot = TranscriptSnapshot {
                    text: text.to_string(),
                    revision: 1,
                    pending_delivery,
                    confidence,
                    words: Vec::new(),
                    recording,
                    expires_at_ms: 0,
                    settings: TranscriptPopupConfig {
                        enabled: true,
                        review_mode: true,
                        ..Default::default()
                    },
                };
                let context = egui::Context::default();
                let output = context.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(760.0, 280.0),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        draw_transcript(ui, 760, 280, &snapshot);
                    },
                );
                let expected = if confidence.is_some() {
                    "Confidence 85/100"
                } else {
                    "Confidence --/100"
                };
                assert!(output.shapes.iter().any(|shape| {
                    matches!(&shape.shape, egui::Shape::Text(text) if text.galley.job.text == expected)
                }));
            }
        }
    }

    #[test]
    fn transcript_popup_review_buttons_deliver_only_ready_text() {
        for pending_delivery in [false, true] {
            for (expected_action, horizontal) in [
                (ReviewAction::Deliver, 72.0),
                (ReviewAction::Discard, 176.0),
            ] {
                let snapshot = TranscriptSnapshot {
                    text: "Reviewed words".to_string(),
                    revision: 1,
                    pending_delivery,
                    confidence: None,
                    words: Vec::new(),
                    recording: false,
                    expires_at_ms: 0,
                    settings: TranscriptPopupConfig {
                        enabled: true,
                        review_mode: true,
                        ..Default::default()
                    },
                };
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
                            action = draw_transcript(ui, 760, 280, &snapshot);
                        },
                    );
                }
                assert_eq!(action, pending_delivery.then_some(expected_action));
            }
        }
    }
}

fn reconfigure_surface(rs: &mut RenderSurface) {
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: rs.surface_format,
        width: rs.width.max(1),
        height: rs.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    rs.surface.configure(&rs.device, &surface_config);
}

fn position_to_anchor_and_margins(pos: OsdPosition, margin: i32) -> (Anchor, i32, i32, i32, i32) {
    // (anchor, top, bottom, left, right)
    match pos {
        OsdPosition::BottomCenter => (Anchor::BOTTOM, 0, margin, 0, 0),
        OsdPosition::TopCenter => (Anchor::TOP, margin, 0, 0, 0),
        OsdPosition::BottomLeft => (Anchor::BOTTOM | Anchor::LEFT, 0, margin, margin, 0),
        OsdPosition::BottomRight => (Anchor::BOTTOM | Anchor::RIGHT, 0, margin, 0, margin),
        OsdPosition::TopLeft => (Anchor::TOP | Anchor::LEFT, margin, 0, margin, 0),
        OsdPosition::TopRight => (Anchor::TOP | Anchor::RIGHT, margin, 0, 0, margin),
    }
}

/// Render the egui UI: scrolling waveform on the left, segmented vertical
/// peak meter on the right.
#[allow(clippy::too_many_arguments)]
fn draw_ui(
    ui: &mut egui::Ui,
    width: u32,
    height: u32,
    palette: &Palette,
    envelope: &[EnvelopeColumn],
    peak_dbfs: f32,
    held_dbfs: f32,
    gain: f32,
) {
    use egui::{Pos2, Rect};
    let painter = ui.painter().clone();
    let w = width as f32;
    let h = height as f32;
    let meter_w = (w * 0.05).max(8.0);
    let waveform_w = w - meter_w - 4.0;
    let waveform_rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(waveform_w, h));
    let meter_rect = Rect::from_min_size(Pos2::new(w - meter_w, 0.0), egui::vec2(meter_w, h));

    draw_waveform(&painter, waveform_rect, palette, envelope, gain);
    draw_meter(&painter, meter_rect, palette, peak_dbfs, held_dbfs);
}

fn draw_waveform(
    painter: &egui::Painter,
    rect: egui::Rect,
    palette: &Palette,
    envelope: &[EnvelopeColumn],
    gain: f32,
) {
    use egui::{pos2, Shape};
    if envelope.is_empty() {
        return;
    }
    let n = envelope.len();
    let col_w = rect.width() / n as f32;
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.45;

    // A waveform is concave. Triangulate each strip explicitly instead of
    // passing the whole outline to egui's convex-polygon tessellator.
    let mut mesh = egui::Mesh::default();
    let fill = color_to_egui(palette.accent);
    let mut top_edge = Vec::with_capacity(n);
    let mut bottom_edge = Vec::with_capacity(n);
    for (i, col) in envelope.iter().enumerate() {
        let x = rect.left() + (i as f32 + 0.5) * col_w;
        let top = mid_y - (col.max * gain).clamp(-1.0, 1.0) * half_h;
        let bot = mid_y - (col.min * gain).clamp(-1.0, 1.0) * half_h;
        mesh.colored_vertex(pos2(x, top), fill);
        mesh.colored_vertex(pos2(x, bot), fill);
        top_edge.push(pos2(x, top));
        bottom_edge.push(pos2(x, bot));
        if i > 0 {
            let v = (i * 2) as u32;
            mesh.add_triangle(v - 2, v - 1, v);
            mesh.add_triangle(v, v - 1, v + 1);
        }
    }
    painter.add(Shape::mesh(mesh));
    // Mesh edges are not antialiased by egui. Its stroked paths keep the
    // silhouette smooth as it moves through fractional pixel positions.
    painter.add(Shape::line(top_edge, egui::Stroke::new(1.0_f32, fill)));
    painter.add(Shape::line(bottom_edge, egui::Stroke::new(1.0_f32, fill)));

    // Centerline tick for visual reference at low levels.
    let line_color = color_to_egui(palette.foreground.with_alpha(0.25));
    painter.line_segment(
        [pos2(rect.left(), mid_y), pos2(rect.right(), mid_y)],
        egui::Stroke::new(1.0_f32, line_color),
    );
}

fn draw_meter(
    painter: &egui::Painter,
    rect: egui::Rect,
    palette: &Palette,
    peak_dbfs: f32,
    held_dbfs: f32,
) {
    use egui::{pos2, Rect};
    const SEGMENTS: usize = 10;
    const FLOOR_DBFS: f32 = -60.0;

    let segment_h = rect.height() / SEGMENTS as f32;
    let segment_gap = (segment_h * 0.15).clamp(1.0, 3.0);
    let inner_w = rect.width() - 4.0;
    let lit_fraction = peak_meter_fraction(peak_dbfs, FLOOR_DBFS);
    let lit_segments = (lit_fraction * SEGMENTS as f32).round() as usize;

    for i in 0..SEGMENTS {
        // Segment 0 is the BOTTOM of the bar (low dB == bottom).
        let y_top = rect.bottom() - (i as f32 + 1.0) * segment_h + segment_gap * 0.5;
        let y_bot = rect.bottom() - i as f32 * segment_h - segment_gap * 0.5;
        let seg_rect = Rect::from_min_max(
            pos2(rect.left() + 2.0, y_top),
            pos2(rect.left() + 2.0 + inner_w, y_bot),
        );

        let segment_peak_dbfs = FLOOR_DBFS * (1.0 - i as f32 / SEGMENTS as f32);
        let zone = MeterZone::from_dbfs(segment_peak_dbfs);
        let lit = i < lit_segments;
        let base = zone.color(palette);
        let color = if lit {
            color_to_egui(base)
        } else {
            color_to_egui(base.with_alpha(0.18))
        };
        painter.rect_filled(seg_rect, 1.0, color);
    }

    // Held-peak tick, drawn as a thin foreground bar.
    let held_fraction = peak_meter_fraction(held_dbfs, FLOOR_DBFS);
    if held_fraction > 0.0 {
        let y = rect.bottom() - held_fraction * rect.height();
        let tick_rect = Rect::from_min_max(
            pos2(rect.left() + 2.0, y - 1.0),
            pos2(rect.left() + 2.0 + inner_w, y + 1.0),
        );
        painter.rect_filled(tick_rect, 0.0, color_to_egui(palette.foreground));
    }
}

fn color_to_egui(c: voxtype::osd::visual::Color) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (c.r.clamp(0.0, 1.0) * 255.0) as u8,
        (c.g.clamp(0.0, 1.0) * 255.0) as u8,
        (c.b.clamp(0.0, 1.0) * 255.0) as u8,
        (c.a.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

// ---------------------------------------------------------------------------
// SCTK delegate trait impls
// ---------------------------------------------------------------------------

impl CompositorHandler for App {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _time: u32,
    ) {
        if let Some(rs) = self.surface.as_mut() {
            if rs.wl_surface != *surface {
                return;
            }
            rs.frame_pending = false;
            self.tick();
        }
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl SeatHandler for App {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer
            && !self.pointers.iter().any(|(current, _)| current == &seat)
        {
            match self.seat_state.get_pointer(qh, &seat) {
                Ok(pointer) => self.pointers.push((seat, pointer)),
                Err(error) => tracing::warn!("Popup pointer unavailable: {error}"),
            }
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer {
            self.pointers.retain(|(current, pointer)| {
                if current == &seat {
                    pointer.release();
                    false
                } else {
                    true
                }
            });
        }
    }

    fn remove_seat(&mut self, conn: &Connection, qh: &QueueHandle<Self>, seat: wl_seat::WlSeat) {
        self.remove_capability(conn, qh, seat, Capability::Pointer);
    }
}

impl PointerHandler for App {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            if self
                .surface
                .as_ref()
                .is_none_or(|surface| surface.wl_surface != event.surface)
            {
                continue;
            }
            let position = egui::pos2(event.position.0 as f32, event.position.1 as f32);
            match event.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
                    self.pointer_events
                        .push(egui::Event::PointerMoved(position));
                }
                PointerEventKind::Leave { .. } => {
                    self.pointer_events.push(egui::Event::PointerGone)
                }
                PointerEventKind::Press { button: 0x110, .. }
                | PointerEventKind::Release { button: 0x110, .. } => {
                    self.pointer_events.push(egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed: matches!(event.kind, PointerEventKind::Press { .. }),
                        modifiers: Default::default(),
                    });
                }
                PointerEventKind::Axis {
                    horizontal,
                    vertical,
                    ..
                } => {
                    self.pointer_events.push(egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::vec2(-horizontal.absolute as f32, -vertical.absolute as f32),
                        phase: egui::TouchPhase::Move,
                        modifiers: Default::default(),
                    });
                }
                _ => {}
            }
            self.transcript_dirty = true;
        }
        if let Err(error) = self.render_frame() {
            tracing::warn!("Transcript pointer render failed: {error}");
        }
    }
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for App {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, layer: &LayerSurface) {
        if let Some(rs) = self.surface.as_ref() {
            if rs.layer.wl_surface().id() == layer.wl_surface().id() {
                tracing::info!("Compositor closed the layer surface");
                self.tear_down_surface();
            }
        }
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let rs = match self.surface.as_mut() {
            Some(s) => s,
            None => return,
        };
        if rs.layer.wl_surface().id() != layer.wl_surface().id() {
            return;
        }
        let (mut w, mut h) = configure.new_size;
        if w == 0 {
            w = self.shared.config.width_px;
        }
        if h == 0 {
            h = self.shared.config.height_px;
        }
        rs.width = w;
        rs.height = h;
        rs.configured = true;
        reconfigure_surface(rs);
        if let Err(e) = self.render_frame() {
            tracing::warn!("initial render after configure failed: {:#}", e);
        }
    }
}

delegate_compositor!(App);
delegate_output!(App);
delegate_layer!(App);
delegate_seat!(App);
delegate_pointer!(App);

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

delegate_registry!(App);
