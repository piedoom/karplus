use std::{ops::RangeInclusive, time::Duration};

use bevy::prelude::*;
use bevy_seedling::prelude::{MainBus, VolumeNode};
use firewheel::Volume;

use crate::{Dripper, InsertMode, audio::AdsrEnvelopeNode};

pub(crate) struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_buttons)
            .add_systems(Update, button_system);
    }
}

const WINDOW_MARGIN: Vec2 = Vec2::splat(16.0);
const PADDING: UiRect = UiRect::all(Val::Px(12.0));
const NORMAL_BUTTON: Color = Color::srgb(0.15, 0.15, 0.15);
const HOVERED_BUTTON: Color = Color::srgb(0.25, 0.25, 0.25);
const PRESSED_BUTTON: Color = Color::srgb(0.35, 0.35, 0.35);
const COLUMN_GAP: Val = Val::Px(16.0);
const TEMPO_DELTA: Duration = Duration::from_millis(200);
const RATE_BOUNDS_MILLIS: RangeInclusive<i32> = 400..=5000;

fn spawn_buttons(mut cmd: Commands) {
    let text = |s: &str| {
        (
            Text::new(s),
            TextFont {
                font_size: 18.,
                ..default()
            },
            TextColor(Color::WHITE),
        )
    };
    cmd.spawn(Node {
        position_type: PositionType::Absolute,
        bottom: Val::Px(WINDOW_MARGIN.y),
        left: Val::Px(WINDOW_MARGIN.x),
        flex_direction: FlexDirection::Row,
        column_gap: COLUMN_GAP,
        ..default()
    })
    .with_children(|cmd| {
        let button_bundle = || {
            (
                Node {
                    padding: PADDING,
                    ..default()
                },
                BorderRadius::all(Val::Px(4.)),
                Button,
            )
        };

        cmd.spawn((button_bundle(), Muted::default()))
            .with_children(|cmd| {
                cmd.spawn(text("Mute"));
            })
            .observe(on_mute_toggle);

        cmd.spawn(button_bundle())
            .with_children(|cmd| {
                cmd.spawn(text("Clear"));
            })
            .observe(on_clear);

        cmd.spawn(Node {
            flex_direction: FlexDirection::Row,
            ..default()
        })
        .with_children(|cmd| {
            cmd.spawn(button_bundle())
                .with_children(|cmd| {
                    cmd.spawn(text("-"));
                })
                .observe(on_change_tempo_millis(TEMPO_DELTA.as_millis() as i32));

            cmd.spawn(button_bundle())
                .with_children(|cmd| {
                    cmd.spawn(text("+"));
                })
                .observe(on_change_tempo_millis(-(TEMPO_DELTA.as_millis() as i32)));
        });

        cmd.spawn(Node {
            flex_direction: FlexDirection::Row,
            ..default()
        })
        .with_children(|cmd| {
            cmd.spawn(button_bundle())
                .with_children(|cmd| {
                    cmd.spawn(text("k"));
                })
                .observe(on_change_mode(InsertMode::Karplus));

            cmd.spawn(button_bundle())
                .with_children(|cmd| {
                    cmd.spawn(text("m"));
                })
                .observe(on_change_mode(InsertMode::Modal));
        });
    });
}

#[derive(Component, Deref, DerefMut, Default)]
struct Muted(pub(crate) bool);

pub fn on_change_mode(
    mode: InsertMode,
) -> impl IntoSystem<Trigger<'static, Pointer<Click>>, (), ()> {
    IntoSystem::into_system(
        move |_trigger: Trigger<Pointer<Click>>, mut existing_mode: ResMut<InsertMode>| {
            *existing_mode = mode;
        },
    )
}

pub fn on_change_tempo_millis(
    delta: i32,
) -> impl IntoSystem<Trigger<'static, Pointer<Click>>, (), ()> {
    IntoSystem::into_system(
        move |_trigger: Trigger<Pointer<Click>>, mut drippers: Query<&mut Dripper>| {
            for mut dripper in drippers.iter_mut() {
                let millis = dripper.interval.as_millis() as i32;
                let new_interval = millis.checked_add(delta).unwrap_or_default().clamp(
                    RATE_BOUNDS_MILLIS.min().unwrap(),
                    RATE_BOUNDS_MILLIS.max().unwrap(),
                );

                dripper.interval = Duration::from_millis(new_interval as u64);
            }
        },
    )
}

fn on_clear(
    _trigger: Trigger<Pointer<Click>>,
    mut cmd: Commands,
    rectangles: Query<Entity, With<AdsrEnvelopeNode>>,
) {
    for r in rectangles.iter() {
        cmd.entity(r).despawn();
    }
}

fn on_mute_toggle(
    trigger: Trigger<Pointer<Click>>,
    children: Query<&Children>,
    mut muted: Single<&mut Muted>,
    mut main: Single<&mut VolumeNode, With<MainBus>>,
    mut last_volume: Local<f32>,
    mut text_query: Query<&mut Text>,
) {
    let children = children.get(trigger.target()).unwrap();
    // Swap mute
    ***muted = !muted.0;
    let mut text = text_query.get_mut(children[0]).unwrap();
    **text = match ***muted {
        true => "Unmute",
        false => "Mute",
    }
    .to_string();

    let new = *last_volume;
    let old = main.volume.linear();
    *last_volume = old;
    main.volume = Volume::Linear(new);
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<Button>),
    >,
) {
    for (interaction, mut color) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = PRESSED_BUTTON.into();
            }
            Interaction::Hovered => {
                *color = HOVERED_BUTTON.into();
            }
            Interaction::None => {
                *color = NORMAL_BUTTON.into();
            }
        }
    }
}
