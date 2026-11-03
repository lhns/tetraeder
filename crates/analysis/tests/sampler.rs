//! The box sampler's output (scripts/detector_sample.py `observe`) parses as an `Observation`
//! and drives the detector (hardware: none). The fixture comes from a synthetic harness run with
//! half the CPU quota busy elsewhere and 6 GB of VRAM held by another process.

use tetraeder_analysis::detect::{FaultClass, detect};
use tetraeder_analysis::observation::Observation;

#[test]
fn sampler_observation_parses_and_names_both_faults() {
    let text = include_str!("../fixtures/sampler_observation.json");
    let obs: Observation = serde_json::from_str(text).unwrap();
    assert_eq!(obs.windows.len(), 4);
    let r = detect(&obs);
    let classes: Vec<FaultClass> = r.faults.iter().map(|f| f.class).collect();
    assert!(classes.contains(&FaultClass::CpuCotenant), "{r:#?}");
    assert!(classes.contains(&FaultClass::VramPressure), "{r:#?}");
}
