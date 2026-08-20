Require Import List.

Import ListNotations.

Section ForwardModelLaws.

Variable Wave ScanPos PotentialSlice DetectorOutput Output : Type.

Variable exit_wave : list PotentialSlice -> Wave.
Variable scanned_exit_waves : list ScanPos -> list PotentialSlice -> list Wave.
Variable detector : Wave -> DetectorOutput.
Variable average_intensity : list Wave -> DetectorOutput.
Variable apply_source_offsets : Wave -> list Wave.
Variable stack_detector_outputs : list DetectorOutput -> Output.
Variable detector_output_to_output : DetectorOutput -> Output.
Variable single_wave_to_output : Wave -> Output.
Variable scanned_waves_to_output : list Wave -> Output.

Definition forward_tem
  (detector_opt : option (Wave -> DetectorOutput))
  (coherence : bool)
  (potential_slices : list PotentialSlice) : Output :=
  let wave := exit_wave potential_slices in
  match coherence with
  | true => detector_output_to_output (average_intensity (apply_source_offsets wave))
  | false =>
      match detector_opt with
      | Some detector_fn => detector_output_to_output (detector_fn wave)
      | None => single_wave_to_output wave
      end
  end.

Definition forward_stem
  (detector_opt : option (Wave -> DetectorOutput))
  (coherence : bool)
  (positions : list ScanPos)
  (potential_slices : list PotentialSlice) : Output :=
  let waves := scanned_exit_waves positions potential_slices in
  match coherence with
  | true =>
      stack_detector_outputs
        (map (fun wave => average_intensity (apply_source_offsets wave)) waves)
  | false =>
      match detector_opt with
      | Some detector_fn => stack_detector_outputs (map detector_fn waves)
      | None => scanned_waves_to_output waves
      end
  end.

Lemma forward_tem_without_scan_detector_or_coherence :
  forall potential_slices,
    forward_tem None false potential_slices =
    single_wave_to_output (exit_wave potential_slices).
Proof.
  reflexivity.
Qed.

Lemma forward_tem_with_detector_no_coherence :
  forall detector_fn potential_slices,
    forward_tem (Some detector_fn) false potential_slices =
    detector_output_to_output (detector_fn (exit_wave potential_slices)).
Proof.
  reflexivity.
Qed.

Lemma forward_tem_with_coherence :
  forall detector_opt potential_slices,
    forward_tem detector_opt true potential_slices =
    detector_output_to_output
      (average_intensity (apply_source_offsets (exit_wave potential_slices))).
Proof.
  reflexivity.
Qed.

Lemma forward_stem_without_detector_or_coherence :
  forall positions potential_slices,
    forward_stem None false positions potential_slices =
    scanned_waves_to_output (scanned_exit_waves positions potential_slices).
Proof.
  reflexivity.
Qed.

Lemma forward_stem_with_detector_no_coherence :
  forall detector_fn positions potential_slices,
    forward_stem (Some detector_fn) false positions potential_slices =
    stack_detector_outputs
      (map detector_fn (scanned_exit_waves positions potential_slices)).
Proof.
  reflexivity.
Qed.

Lemma forward_stem_with_coherence :
  forall detector_opt positions potential_slices,
    forward_stem detector_opt true positions potential_slices =
    stack_detector_outputs
      (map
         (fun wave => average_intensity (apply_source_offsets wave))
         (scanned_exit_waves positions potential_slices)).
Proof.
  reflexivity.
Qed.

End ForwardModelLaws.
