use bevy::{color::palettes::css::RED, prelude::*};
use bevy_seedling::prelude::{MainBus, VolumeNode};
use firewheel::Volume;

use crate::audio::AdsrEnvelopeNode;

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

fn spawn_buttons(mut cmd: Commands) {
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
                cmd.spawn((
                    Text::new("Mute"),
                    TextFont {
                        font_size: 18.,
                        ..default()
                    },
                    TextColor(Color::WHITE),
                ));
            })
            .observe(on_mute_toggle);

        cmd.spawn(button_bundle())
            .with_children(|cmd| {
                cmd.spawn((
                    Text::new("Clear"),
                    TextFont {
                        font_size: 18.,
                        ..default()
                    },
                    TextColor(Color::WHITE),
                ));
            })
            .observe(on_clear);
    });
}

#[derive(Component, Deref, DerefMut, Default)]
struct Muted(pub(crate) bool);

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
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (Changed<Interaction>, With<Button>),
    >,
) {
    for (interaction, mut color, mut border_color) in &mut interaction_query {
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
