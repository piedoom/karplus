//! Pink noise is taken from firewhell since the crates aren't updated at the
//! time of writing A simple node that generates pink noise.
//!
//! Base on the algorithm from <https://www.musicdsp.org/en/latest/Synthesis/244-direct-pink-noise-synthesis-with-auto-correlated-generator.html>

use bevy::prelude::*;
use bevy_seedling::prelude::NodeLabel;
pub(crate) use comb::CombNode;
pub(crate) use envelope::AdsrEnvelopeNode;
use firewheel::core::{
    channel_config::{ChannelConfig, ChannelCount},
    diff::{Diff, Patch},
    dsp::{
        filter::smoothing_filter::DEFAULT_SMOOTH_SECONDS,
        volume::{DEFAULT_AMP_EPSILON, Volume},
    },
    event::ProcEvents,
    node::{
        AudioNode, AudioNodeInfo, AudioNodeProcessor, ConstructProcessorContext, ProcBuffers,
        ProcExtra, ProcInfo, ProcessStatus,
    },
    param::smoother::{SmoothedParam, SmootherConfig},
};
pub(crate) use math::{MathNode, Operation};
pub(crate) use pink_noise::PinkNoiseGenNode;

mod pink_noise {
    use super::*;

    const COEFF_A: [i32; 5] = [14055, 12759, 10733, 12273, 15716];
    const COEFF_SUM: [i16; 5] = [22347, 27917, 29523, 29942, 30007];

    /// A simple node that generates white noise. (Mono output only)
    #[derive(Component, Diff, Patch, Debug, Clone, Copy, PartialEq)]
    pub struct PinkNoiseGenNode {
        /// The overall volume.
        ///
        /// Note, pink noise is really loud, so prefer to use a value like
        /// `Volume::Linear(0.4)` or `Volume::Decibels(-18.0)`.
        pub volume: Volume,
        /// Whether or not this node is enabled.
        pub enabled: bool,
        /// The time in seconds of the internal smoothing filter.
        ///
        /// By default this is set to `0.015` (15ms).
        pub smooth_seconds: f32,
    }

    impl Default for PinkNoiseGenNode {
        fn default() -> Self {
            Self {
                volume: Volume::Linear(0.4),
                enabled: true,
                smooth_seconds: DEFAULT_SMOOTH_SECONDS,
            }
        }
    }

    /// The configuration for a [`PinkNoiseGenNode`]
    #[derive(Component, Debug, Clone, PartialEq)]
    pub struct PinkNoiseGenConfig {
        /// The starting seed. This cannot be zero.
        pub seed: i32,
    }

    impl Default for PinkNoiseGenConfig {
        fn default() -> Self {
            Self { seed: 17 }
        }
    }

    impl AudioNode for PinkNoiseGenNode {
        type Configuration = PinkNoiseGenConfig;

        fn info(&self, _config: &Self::Configuration) -> AudioNodeInfo {
            AudioNodeInfo::new()
                .debug_name("pink_noise_gen")
                .channel_config(ChannelConfig {
                    num_inputs: ChannelCount::ZERO,
                    num_outputs: ChannelCount::MONO,
                })
        }

        fn construct_processor(
            &self,
            config: &Self::Configuration,
            cx: ConstructProcessorContext,
        ) -> impl AudioNodeProcessor {
            // Seed cannot be zero.
            let seed = if config.seed == 0 { 17 } else { config.seed };

            Processor {
                gain: SmoothedParam::new(
                    self.volume.amp_clamped(DEFAULT_AMP_EPSILON),
                    SmootherConfig {
                        smooth_seconds: self.smooth_seconds,
                        ..Default::default()
                    },
                    cx.stream_info.sample_rate,
                ),
                params: *self,
                fpd: seed,
                contrib: [0; 5],
                accum: 0,
            }
        }
    }

    // The realtime processor counterpart to your node.
    struct Processor {
        params: PinkNoiseGenNode,
        gain: SmoothedParam,

        // white noise generator state
        fpd: i32,

        // filter stage contributions
        contrib: [i32; 5],
        accum: i32,
    }

    impl AudioNodeProcessor for Processor {
        fn process(
            &mut self,
            info: &ProcInfo,
            buffers: ProcBuffers,
            events: &mut ProcEvents,
            _extra: &mut ProcExtra,
        ) -> ProcessStatus {
            for patch in events.drain_patches::<PinkNoiseGenNode>() {
                match patch {
                    PinkNoiseGenNodePatch::Volume(vol) => {
                        self.gain.set_value(vol.amp_clamped(DEFAULT_AMP_EPSILON));
                    }
                    PinkNoiseGenNodePatch::SmoothSeconds(seconds) => {
                        self.gain.set_smooth_seconds(seconds, info.sample_rate);
                    }
                    _ => {}
                }

                self.params.apply(patch);
            }

            if !self.params.enabled
                || (self.gain.target_value() == 0.0 && !self.gain.is_smoothing())
            {
                self.gain.reset();
                return ProcessStatus::ClearAllOutputs;
            }

            for s in buffers.outputs[0].iter_mut() {
                // i16[0,32767]
                let randu: i16 = (rng(&mut self.fpd) & 0x7fff) as i16;

                // i32[-32768,32767]
                let r_bytes = rng(&mut self.fpd).to_ne_bytes();
                let randv: i32 = i16::from_ne_bytes([r_bytes[0], r_bytes[1]]) as i32;

                if randu < COEFF_SUM[0] {
                    update_contrib::<0>(&mut self.accum, &mut self.contrib, randv);
                } else if randu < COEFF_SUM[1] {
                    update_contrib::<1>(&mut self.accum, &mut self.contrib, randv);
                } else if randu < COEFF_SUM[2] {
                    update_contrib::<2>(&mut self.accum, &mut self.contrib, randv);
                } else if randu < COEFF_SUM[3] {
                    update_contrib::<3>(&mut self.accum, &mut self.contrib, randv);
                } else if randu < COEFF_SUM[4] {
                    update_contrib::<4>(&mut self.accum, &mut self.contrib, randv);
                }

                // Get a random normalized value in the range `[-1.0, 1.0]`.
                let r = self.accum as f32 * (1.0 / 2_147_483_648.0);

                *s = r * self.gain.next_smoothed();
            }

            ProcessStatus::outputs_not_silent()
        }
    }

    #[inline(always)]
    fn rng(fpd: &mut i32) -> i32 {
        *fpd ^= *fpd << 13;
        *fpd ^= *fpd >> 17;
        *fpd ^= *fpd << 5;

        *fpd
    }

    #[inline(always)]
    fn update_contrib<const I: usize>(accum: &mut i32, contrib: &mut [i32; 5], randv: i32) {
        *accum = accum.wrapping_sub(contrib[I]);
        contrib[I] = randv * COEFF_A[I];
        *accum = accum.wrapping_add(contrib[I]);
    }
}

mod envelope {

    use firewheel::{
        SilenceMask,
        dsp::filter::smoothing_filter::{SmoothingFilter, SmoothingFilterCoeff},
        event,
    };

    /// The configuration for a [`DummyNode`], a node which does nothing.
    #[derive(Component, Default, Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DummyNodeConfig {
        pub channel_config: ChannelConfig,
    }

    use super::*;
    /// A simple Attack-Decay-Sustain-Release envelope generator. This signal
    /// can be fed into a multiplier with another signal to shape its
    /// output.
    #[derive(Component, Diff, Patch, Debug, Clone, Copy, PartialEq)]
    pub struct AdsrEnvelopeNode {
        /// The attack in seconds
        pub attack: f32,
        /// The decay in seconds
        pub decay: f32,
        /// The amplitude of the sustain
        pub sustain: Volume,
        /// The release in seconds
        pub release: f32,
        /// The incoming gate signal. Use this to generate an envelope signal.
        pub gate: bool,
        pub velocity: f32,
    }

    impl Default for AdsrEnvelopeNode {
        fn default() -> Self {
            Self {
                attack: 0.005,
                decay: 0.01,
                sustain: Volume::Linear(0.7),
                release: 0.05,
                velocity: 1.0,
                gate: false,
            }
        }
    }

    enum State {
        Idling,
        Attacking,
        Decaying,
        Sustaining,
        Releasing,
    }

    impl AudioNode for AdsrEnvelopeNode {
        type Configuration = DummyNodeConfig;

        fn info(&self, _config: &Self::Configuration) -> AudioNodeInfo {
            AudioNodeInfo::new()
                .debug_name("adsr_envelope_generator")
                .channel_config(ChannelConfig {
                    num_inputs: ChannelCount::ZERO,
                    num_outputs: ChannelCount::MONO,
                })
        }

        fn construct_processor(
            &self,
            _config: &Self::Configuration,
            cx: ConstructProcessorContext,
        ) -> impl AudioNodeProcessor {
            let smooth = |seconds| {
                (
                    SmoothingFilterCoeff::new(cx.stream_info.sample_rate, seconds),
                    SmoothingFilter::new(0.0),
                )
            };
            Processor {
                state: State::Idling,
                attack: smooth(self.attack),
                decay: smooth(self.decay),
                sustain: self.sustain,
                release: smooth(self.release),
                enabled: true,
                current_level: 0f32,
                velocity: 0.0,
            }
        }
    }

    struct Processor {
        state: State,
        attack: (SmoothingFilterCoeff, SmoothingFilter),
        decay: (SmoothingFilterCoeff, SmoothingFilter),
        sustain: Volume,
        release: (SmoothingFilterCoeff, SmoothingFilter),
        enabled: bool,
        current_level: f32,
        velocity: f32,
    }

    impl AudioNodeProcessor for Processor {
        fn process(
            &mut self,
            info: &ProcInfo,
            buffers: ProcBuffers,
            events: &mut event::ProcEvents,
            _extra: &mut ProcExtra,
        ) -> ProcessStatus {
            let smooth = |seconds| SmoothingFilterCoeff::new(info.sample_rate, seconds);
            use AdsrEnvelopeNodePatch::*;
            for patch in events.drain_patches::<AdsrEnvelopeNode>() {
                match patch {
                    Attack(attack) => self.attack.0 = smooth(attack),
                    Decay(decay) => self.decay.0 = smooth(decay),
                    Sustain(sustain) => self.sustain = sustain,
                    Release(release) => self.release.0 = smooth(release),
                    Gate(gate) => {
                        // Derive a new state when the gate signal has changed
                        self.state = match gate {
                            true => {
                                self.attack.1 = SmoothingFilter::new(self.current_level);
                                State::Attacking
                            }
                            false => {
                                self.release.1 = SmoothingFilter::new(self.current_level);
                                State::Releasing
                            }
                        };
                    }
                    Velocity(velocity) => self.velocity = velocity.clamp(0., 1.),
                }
            }

            if matches!(self.state, State::Idling) || !self.enabled {
                buffers.outputs[0].fill(0f32);
                return ProcessStatus::outputs_modified(SilenceMask(0));
            }

            // Mono only
            buffers.outputs[0].iter_mut().for_each(|sample| {
                // Look through each state and exponentially interpolate accordingly
                use State::*;
                let level = match self.state {
                    Idling => 0.0,
                    Attacking => {
                        let (coeff, filter) = &mut self.attack;
                        let level = filter.process(1.0 * self.velocity, *coeff);
                        // Advance if the peak is reached
                        if level > (0.999 * self.velocity) {
                            self.decay.1 = SmoothingFilter::new(self.velocity);
                            self.state = Decaying;
                        }
                        level
                    }
                    Decaying => {
                        let (coeff, filter) = &mut self.decay;
                        let level = filter.process(self.sustain.linear() * self.velocity, *coeff);
                        if level <= self.sustain.linear() * self.velocity {
                            self.state = Sustaining;
                        }
                        level
                    }
                    Sustaining => self.sustain.linear(),
                    Releasing => {
                        let (coeff, filter) = &mut self.release;
                        let level = filter.process(0.0, *coeff);
                        if level < 0.001 {
                            self.state = Idling;
                        }
                        level
                    }
                };
                *sample = level;
            });

            self.current_level = *buffers.outputs[0].last().unwrap();

            ProcessStatus::outputs_not_silent()
        }
    }
}

mod comb {
    use std::array::from_fn;

    use firewheel::event;

    use super::*;

    /// A comb filter with simple cutoff
    #[derive(Component, Diff, Patch, Debug, Clone, Copy, PartialEq)]
    pub struct CombNode<const CHANNELS: usize> {
        /// The delay offset in samples (always negative)
        pub delay: u16,
        /// Whether or not this node is enabled.
        pub enabled: bool,
        /// Amount of feedback to return from the delay
        pub feedback: Volume,
        /// The cutoff represented as a linear value, where 1.0 is no filtering
        /// and 0.0 is maximum filtering.
        pub cutoff: f32,
    }

    /// Configuration for a [`CombNode`].
    #[derive(Component, PartialEq, Clone)]
    pub struct CombNodeConfig {
        /// The maximum number of samples stored, dictating the maximum length
        /// of delay.
        pub buffer_size: usize,
    }

    impl Default for CombNodeConfig {
        fn default() -> Self {
            // 100ms of buffer time
            Self { buffer_size: 4800 }
        }
    }

    impl<const CHANNELS: usize> Default for CombNode<CHANNELS> {
        fn default() -> Self {
            Self {
                delay: 1,
                enabled: true,
                feedback: Volume::Decibels(-1f32),
                cutoff: 0.1,
            }
        }
    }

    impl<const CHANNELS: usize> AudioNode for CombNode<CHANNELS> {
        type Configuration = CombNodeConfig;

        fn info(&self, _config: &Self::Configuration) -> AudioNodeInfo {
            AudioNodeInfo::new()
                .debug_name("comb_filter")
                .channel_config(ChannelConfig {
                    // There's not a great reason to allow this to be multi-channel yet, since the
                    // left/right parts do not interact.
                    num_inputs: ChannelCount::new(CHANNELS as u32).unwrap(),
                    num_outputs: ChannelCount::new(CHANNELS as u32).unwrap(),
                })
        }

        // Construct the realtime processor counterpart using the given information
        // about the audio stream.
        //
        // This method is called before the node processor is sent to the realtime
        // thread, so it is safe to do non-realtime things here like allocating.
        fn construct_processor(
            &self,
            config: &Self::Configuration,
            _cx: ConstructProcessorContext,
        ) -> impl AudioNodeProcessor {
            Processor::<CHANNELS> {
                buffers: from_fn(|_| vec![0f32; config.buffer_size]),
                feedback: self.feedback.linear(),
                enabled: true,
                delay: self.delay,
                cursors: [0; CHANNELS],
                cutoff_z_neg_one: [0.0; CHANNELS],
                cutoff: 0.1,
            }
        }
    }

    struct Processor<const CHANNELS: usize> {
        /// Audio buffers for each channel from which to apply delay and
        /// feedback
        buffers: [Vec<f32>; CHANNELS],
        /// Contains the write head of the circular buffer for each channel
        cursors: [usize; CHANNELS],
        /// The amount of feedback as linear amplitude
        feedback: f32,
        /// The number of samples to delay the signal
        delay: u16,
        enabled: bool,
        /// An additional buffer containing the last rendered sample of every
        /// channel. This is absolutely *not* necessary, as all of this
        /// information is already in our buffers. However, I'm lazy and didn't
        /// feel like refactoring this.
        cutoff_z_neg_one: [f32; CHANNELS],
        /// The scalar of our feedback
        cutoff: f32,
    }

    impl<const CHANNELS: usize> AudioNodeProcessor for Processor<CHANNELS> {
        fn process(
            &mut self,
            _info: &ProcInfo,
            buffers: ProcBuffers,
            events: &mut event::ProcEvents,
            _extra: &mut ProcExtra,
        ) -> ProcessStatus {
            for patch in events.drain_patches::<CombNode<CHANNELS>>() {
                match patch {
                    CombNodePatch::Delay(delay) => self.delay = delay,
                    CombNodePatch::Enabled(enabled) => self.enabled = enabled,
                    CombNodePatch::Feedback(feedback) => self.feedback = feedback.linear(),
                    CombNodePatch::Cutoff(alpha) => self.cutoff = alpha,
                }
            }

            if !self.enabled {
                return ProcessStatus::Bypass;
            }

            // Inputs and outputs are always symmetric, so we can zip
            buffers
                .inputs
                .iter()
                .zip(buffers.outputs.iter_mut())
                .enumerate()
                .for_each(|(channel, (input, output))| {
                    input
                        .iter()
                        .zip(output.iter_mut())
                        .for_each(|(input, output)| {
                            let buffer = &mut self.buffers[channel];
                            let cursor = &mut self.cursors[channel];
                            let delay_output = &mut buffer[*cursor];

                            *output = (input + *delay_output) / CHANNELS as f32;

                            // Add feedback
                            let feedback = input + (*delay_output * self.feedback);

                            // Simple low pass
                            let z = self.cutoff_z_neg_one[channel];
                            let y = self.cutoff * feedback + (1.0 - self.cutoff) * z;
                            self.cutoff_z_neg_one[channel] = y;

                            *delay_output = y;

                            *cursor = (*cursor + 1) % self.delay as usize;
                        });
                });
            ProcessStatus::outputs_not_silent()
        }
    }
}

mod math {
    //! Basic math operation nodes

    use firewheel::event;

    use super::*;
    use crate::audio::envelope::DummyNodeConfig;

    /// An operation, represented as an enum. This approach, as opposed to
    /// separate nodes, allows for easily swapping operations without
    /// adjusting the node graph for greater flexibility.
    #[derive(Diff, Patch, PartialEq, Clone, Copy, Debug)]
    pub enum Operation {
        Add,
        Multiply,
    }

    /// Perform a mathematical operation on one or more signals, defined by
    /// constant `CHANNELS`.
    #[derive(Component, Diff, Patch, Debug, Clone, Copy, PartialEq)]
    pub struct MathNode<const CHANNELS: usize> {
        /// The [`Operation`] to take on each signal
        pub operation: Operation,
    }

    impl<const CHANNELS: usize> Default for MathNode<CHANNELS> {
        fn default() -> Self {
            Self {
                operation: Operation::Add,
            }
        }
    }

    impl<const CHANNELS: usize> AudioNode for MathNode<CHANNELS> {
        type Configuration = DummyNodeConfig;

        fn info(&self, _config: &Self::Configuration) -> AudioNodeInfo {
            AudioNodeInfo::new()
                .debug_name("math_node")
                .channel_config(ChannelConfig {
                    num_inputs: ChannelCount::new(CHANNELS as u32).unwrap(),
                    num_outputs: ChannelCount::MONO,
                })
        }

        fn construct_processor(
            &self,
            _config: &Self::Configuration,
            _cx: ConstructProcessorContext,
        ) -> impl AudioNodeProcessor {
            MathProcessor::<CHANNELS> {
                operation: self.operation,
            }
        }
    }

    struct MathProcessor<const CHANNELS: usize> {
        operation: Operation,
    }

    impl<const CHANNELS: usize> AudioNodeProcessor for MathProcessor<CHANNELS> {
        fn process(
            &mut self,
            info: &ProcInfo,
            buffers: ProcBuffers,
            events: &mut event::ProcEvents,
            _extra: &mut ProcExtra,
        ) -> ProcessStatus {
            for patch in events.drain_patches::<MathNode<CHANNELS>>() {
                match patch {
                    MathNodePatch::Operation(operation) => {
                        self.operation = operation;
                    }
                }
            }

            // Always mono output
            let output = &mut buffers.outputs[0];

            match self.operation {
                Operation::Add =>
                    for i in 0..info.frames {
                        output[i] = buffers
                            .inputs
                            .iter()
                            .fold(0.0, |acc, buffer| acc + buffer[i]);
                    },
                Operation::Multiply =>
                    for i in 0..info.frames {
                        output[i] = buffers
                            .inputs
                            .iter()
                            .fold(1.0, |acc, buffer| acc * buffer[i]);
                    },
            };

            ProcessStatus::outputs_not_silent()
        }
    }
}

#[derive(NodeLabel, PartialEq, Eq, Debug, Hash, Clone)]
pub(crate) struct AppBus;
