// Support configuring Bevy lints within code.
#![cfg_attr(bevy_lint, feature(register_tool), register_tool(bevy))]

pub(crate) mod audio;
mod ui;

use std::{ops::RangeInclusive, time::Duration};

use avian2d::prelude::{
    Collider, CollisionEventsEnabled, Gravity, LinearVelocity, OnCollisionEnd, OnCollisionStart,
    Restitution, RigidBody,
};
use bevy::{
    asset::AssetMetaCheck,
    color::palettes::css::{MAGENTA, WHITE},
    core_pipeline::bloom::Bloom,
    prelude::*,
};
use bevy_seedling::{
    SeedlingPlugin,
    context::SampleRate,
    edge::Connect,
    node::RegisterNode,
    prelude::{FreeverbNode, LowPassNode, VolumeNode},
};
use firewheel::Volume;

use crate::audio::{AdsrEnvelopeNode, AppBus, CombNode, MathNode, Operation, PinkNoiseGenNode};

const GRAVITY: f32 = -30.0;
const RECT_HEIGHT: f32 = 2.5;

fn main() -> AppExit {
    App::new().add_plugins(AppPlugin).run()
}

pub struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        // Add Bevy plugins.
        app.add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    // Wasm builds will check for meta files (that don't exist) if this isn't set.
                    // This causes errors and even panics on web build on itch.
                    // See https://github.com/bevyengine/bevy_github_ci_template/issues/48.
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Window {
                        title: "Karplus".to_string(),
                        fit_canvas_to_parent: true,
                        ..default()
                    }
                    .into(),
                    ..default()
                }),
            MeshPickingPlugin::default(),
            avian2d::PhysicsPlugins::default(),
            SeedlingPlugin::default(),
            ui::UiPlugin,
        ))
        .init_resource::<FirstPoint>()
        .init_resource::<CurrentPoint>()
        .insert_resource(ClearColor(Color::srgb_u8(17, 22, 34)))
        .insert_resource(Gravity(Vec2::Y * GRAVITY))
        .register_node::<audio::AdsrEnvelopeNode>()
        .register_node::<audio::CombNode<1>>()
        .register_node::<audio::MathNode<2>>()
        .register_node::<audio::PinkNoiseGenNode>();

        // Spawn the main camera.
        app.add_systems(Startup, spawn_scene).add_systems(
            Update,
            (
                draw_grid,
                adjust_timers,
                drip,
                draw_rect_preview,
                despawn_balls,
                reset,
            ),
        );
    }
}

fn spawn_scene(mut cmd: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    const DRIPPER_OFFSET: f32 = 48.0;
    // Set up audio channels
    cmd.spawn((VolumeNode::default(), AppBus))
        .chain_node(FreeverbNode {
            room_size: 0.7,
            damping: 0.6,
            width: 0.7,
        });
    // Camera, spawned with a 2.5D style
    cmd.spawn((
        Camera3d::default(),
        Transform::default().with_translation(Vec3::default().with_z(120.)),
        Bloom::OLD_SCHOOL,
    ));
    // The entity that will be creating balls that generate tone
    cmd.spawn((
        Dripper {
            interval: Duration::from_secs(1),
        },
        Transform::from_translation(Vec3::Y * DRIPPER_OFFSET),
    ));
    // The background plane, used for interacting with bevy picking
    cmd.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(1000.)))),
        Pickable {
            should_block_lower: false,
            is_hoverable: true,
        },
        Transform::default(),
    ))
    .observe(on_hover)
    .observe(on_click);
}

fn despawn_balls(mut cmd: Commands, balls: Query<(Entity, &Transform), With<Ball>>) {
    const MIN_HEIGHT: f32 = -40f32;
    for (ball, transform) in balls.iter() {
        if transform.translation.y <= MIN_HEIGHT {
            cmd.entity(ball).despawn();
        }
    }
}

#[derive(Component)]
struct Ball;

// Resource describing our first clicked point, if it exists.
#[derive(Resource, Default, Deref, DerefMut)]
struct FirstPoint(Option<Vec2>);

// The current mouse position in world units
#[derive(Resource, Default, Deref, DerefMut)]
struct CurrentPoint(Option<Vec2>);

fn on_click(
    trigger: Trigger<Pointer<Pressed>>,
    mut cmd: Commands,
    mut first_point_res: ResMut<FirstPoint>,
    current_point: Res<CurrentPoint>,
    mut mesh: Local<Option<Handle<Mesh>>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sample_rate: Res<SampleRate>,
) {
    const COLOR_LENGTH_BOUNDS: RangeInclusive<f32> = 0.0..=40.0;
    const AUDIO_SCALE: f32 = 2000.0;

    fn length_to_color(len: f32) -> Color {
        let (start, end) = COLOR_LENGTH_BOUNDS.into_inner();
        let t = (len.clamp(start, end) - start) / (end - start);
        Color::from(Oklcha::new(1.0, 0.7, 300.0 - (t * 300.0), 1.0))
    }

    if trigger.button == PointerButton::Primary {
        match **first_point_res {
            Some(first_point) => {
                if let Some(current_point) = current_point.0 {
                    // Commit the rectangle
                    let mesh_handle = mesh
                        .get_or_insert_with(|| meshes.add(Cuboid::default()))
                        .clone();

                    let diff = current_point - first_point;
                    let length = diff.length();
                    let angle = diff.y.atan2(diff.x);
                    let mid = (first_point + current_point) * 0.5;

                    cmd.spawn((
                        Ball,
                        Mesh3d(mesh_handle),
                        MeshMaterial3d(
                            materials.add(StandardMaterial::from_color(length_to_color(length))),
                        ),
                        Transform::from_xyz(mid.x, mid.y, 0.0)
                            .with_rotation(Quat::from_rotation_z(angle))
                            .with_scale(Vec3::new(length, RECT_HEIGHT, 1.0)),
                        Collider::rectangle(1.0, 1.0),
                        RigidBody::Static,
                        Restitution::PERFECTLY_ELASTIC,
                        CollisionEventsEnabled,
                        AdsrEnvelopeNode {
                            attack: 0.004,
                            decay: 0.25,
                            sustain: Volume::Linear(0.0),
                            release: 0.15,
                            gate: false,
                            velocity: 0.0,
                        },
                    ))
                    .with_children(|cmd| {
                        // Calculate pitch based on length.
                        // Generator and effects graph
                        let adsr = cmd.target_entity();
                        let noise = cmd.spawn(PinkNoiseGenNode::default()).id();
                        let multiplier = cmd
                            .spawn(MathNode::<2> {
                                operation: Operation::Multiply,
                            })
                            .id();

                        let freq = (1.0 / length) * AUDIO_SCALE;
                        let period = sample_rate.get().get() as f32 / freq;

                        let comb = cmd
                            .spawn((CombNode::<1> {
                                delay: period as u16,
                                feedback: Volume::Linear(0.99),
                                cutoff: 0.69,
                                ..default()
                            },))
                            .id();

                        let lpf = cmd
                            .spawn(LowPassNode {
                                frequency: 1_000f32,
                            })
                            .id();

                        // Attach nodes
                        cmd.commands()
                            .entity(adsr)
                            .connect_with(multiplier, &[(0, 0)]);

                        cmd.commands()
                            .entity(noise)
                            .connect_with(multiplier, &[(0, 1)]);

                        cmd.commands()
                            .entity(multiplier)
                            .connect_with(lpf, &[(0, 0)]);

                        cmd.commands().entity(lpf).connect_with(comb, &[(0, 0)]);

                        cmd.commands()
                            .entity(comb)
                            .connect_with(AppBus, &[(0, 0), (0, 1)]);
                    })
                    .observe(on_hit)
                    .observe(on_hit_end)
                    .observe(on_remove);

                    // Remove the clicked point
                    **first_point_res = None;
                }
            }
            None => {
                // Set the first clicked point
                *first_point_res = FirstPoint(trigger.hit.position.map(|s| s.truncate()));
            }
        }
    }
}

fn on_remove(trigger: Trigger<Pointer<Pressed>>, mut cmd: Commands) {
    if trigger.button == PointerButton::Secondary {
        cmd.entity(trigger.target()).despawn();
    }
}

fn on_hit(
    trigger: Trigger<OnCollisionStart>,
    mut adsrs: Query<&mut AdsrEnvelopeNode>,
    velocities: Query<&LinearVelocity>,
) {
    const VELOCITY_BOUNDS: RangeInclusive<f32> = 0.0..=100.0;
    for mut adsr in adsrs.get_mut(trigger.target()).into_iter() {
        let velocity = velocities.get(trigger.collider).unwrap().length();
        let (start, end) = VELOCITY_BOUNDS.into_inner();
        let impact_strength =
            ((0.1 * (velocity - start) / (end - start)).exp() - 1.0) / ((0.1f32 * 1.0).exp() - 1.0);

        adsr.gate = true;
        adsr.velocity = impact_strength;
    }
}

fn on_hit_end(trigger: Trigger<OnCollisionEnd>, mut adsrs: Query<&mut AdsrEnvelopeNode>) {
    for mut adsr in adsrs.get_mut(trigger.target()).into_iter() {
        adsr.gate = false;
    }
}

fn on_hover(trigger: Trigger<Pointer<Move>>, mut point: ResMut<CurrentPoint>) {
    **point = trigger.hit.position.map(|pos| pos.truncate());
}

fn draw_rect_preview(point_a: Res<FirstPoint>, point_b: Res<CurrentPoint>, mut gizmos: Gizmos) {
    match (point_a.0, point_b.0) {
        (Some(a), Some(b)) => {
            let direction = b - a;
            let length = direction.length();
            let angle = direction.y.atan2(direction.x); // rotation angle in radians
            let midpoint = (a + b) / 2.0;

            gizmos.rect_2d(
                Isometry2d {
                    rotation: angle.into(),
                    translation: midpoint.into(),
                },
                Vec2::new(length, RECT_HEIGHT),
                MAGENTA,
            );
        }
        _ => (),
    }
}

/// Component that spawns in spheres at a specific interval
#[derive(Component)]
#[require(DripTimer)]
struct Dripper {
    pub(crate) interval: Duration,
}

#[derive(Component, DerefMut, Deref)]
struct DripTimer(Timer);

impl Default for DripTimer {
    fn default() -> Self {
        Self(Timer::new(Default::default(), TimerMode::Repeating))
    }
}

fn draw_grid(mut gizmos: Gizmos) {
    const GRID_CELLS: u32 = 100;
    const GRID_SPACING: f32 = 4.0;
    const ALPHA: f32 = 0.02;
    gizmos.grid(
        Isometry3d::default(),
        UVec2::splat(GRID_CELLS),
        Vec2::splat(GRID_SPACING),
        WHITE.with_alpha(ALPHA),
    );
}

fn adjust_timers(changed: Query<(&Dripper, &mut DripTimer), Changed<Dripper>>) {
    for (dripper, mut timer) in changed {
        timer.0.set_duration(dripper.interval);
    }
}

// Create a steady drip of colliders
fn drip(
    mut cmd: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut drippers: Query<(&Transform, &mut DripTimer)>,
    mut mesh: Local<Option<Handle<Mesh>>>,
    mut material: Local<Option<Handle<StandardMaterial>>>,
    time: Res<Time>,
) {
    const RADIUS: f32 = 0.5;
    for (transform, mut timer) in drippers.iter_mut() {
        timer.tick(time.delta());
        if timer.just_finished() {
            let handle = mesh
                .get_or_insert_with(|| meshes.add(Sphere { radius: RADIUS }))
                .clone();
            let material = material
                .get_or_insert_with(|| materials.add(StandardMaterial::from_color(Color::WHITE)))
                .clone();
            cmd.spawn((
                Mesh3d(handle),
                MeshMaterial3d(material),
                transform.clone(),
                RigidBody::Dynamic,
                Collider::circle(RADIUS),
                Restitution::PERFECTLY_ELASTIC,
            ))
            .with_children(|cmd| {
                cmd.spawn((
                    PointLight { ..default() },
                    Transform::from_translation(Vec3::Z),
                ));
            });
        }
    }
}

fn reset(
    keys: Res<ButtonInput<KeyCode>>,
    mut cmd: Commands,
    rectangles: Query<Entity, With<AdsrEnvelopeNode>>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        for entity in rectangles.iter() {
            cmd.entity(entity).despawn();
        }
    }
}
