Require Import List.

Import ListNotations.

Section ForwardEquationCompositionLaws.

Variable SourceWave Wave DetectorOutput Output : Type.

Variable source_wave : SourceWave.
Variable source_tilt : SourceWave -> SourceWave.
Variable source_position : SourceWave -> SourceWave.
Variable scan : SourceWave -> list SourceWave.
Variable interaction : SourceWave -> Wave.
Variable coherence : Wave -> Output.
Variable detector : Wave -> Output.
Variable postprocess : Output -> Output.
Variable stack_outputs : list Output -> Output.
Variable waves_to_output : list Wave -> Output.
Variable wave_to_output : Wave -> Output.

Definition hrtem_forward
  (use_source_tilt use_source_position use_coherence use_detector use_postprocess : bool)
  : Output :=
  let source1 := if use_source_tilt then source_tilt source_wave else source_wave in
  let source2 := if use_source_position then source_position source1 else source1 in
  let wave := interaction source2 in
  let output :=
    if use_coherence then coherence wave
    else if use_detector then detector wave
    else wave_to_output wave in
  if use_postprocess then postprocess output else output.

Definition stem_forward
  (use_source_tilt use_coherence use_detector use_postprocess : bool)
  : Output :=
  let source1 := if use_source_tilt then source_tilt source_wave else source_wave in
  let scanned := scan source1 in
  let waves := map interaction scanned in
  let output :=
    if use_coherence then stack_outputs (map coherence waves)
    else if use_detector then stack_outputs (map detector waves)
    else waves_to_output waves in
  if use_postprocess then postprocess output else output.

Lemma hrtem_forward_plain_wave :
  hrtem_forward false false false false false =
  wave_to_output (interaction source_wave).
Proof.
  reflexivity.
Qed.

Lemma hrtem_forward_without_source_tilt_or_position :
  forall use_coherence use_detector use_postprocess,
    hrtem_forward false false use_coherence use_detector use_postprocess =
    let wave := interaction source_wave in
    let output :=
      if use_coherence then coherence wave
      else if use_detector then detector wave
      else wave_to_output wave in
    if use_postprocess then postprocess output else output.
Proof.
  intros use_coherence use_detector use_postprocess.
  destruct use_coherence, use_detector, use_postprocess; reflexivity.
Qed.

Lemma hrtem_forward_source_tilt_only :
  forall use_coherence use_detector use_postprocess,
    hrtem_forward true false use_coherence use_detector use_postprocess =
    let wave := interaction (source_tilt source_wave) in
    let output :=
      if use_coherence then coherence wave
      else if use_detector then detector wave
      else wave_to_output wave in
    if use_postprocess then postprocess output else output.
Proof.
  intros use_coherence use_detector use_postprocess.
  destruct use_coherence, use_detector, use_postprocess; reflexivity.
Qed.

Lemma hrtem_forward_source_position_only :
  forall use_coherence use_detector use_postprocess,
    hrtem_forward false true use_coherence use_detector use_postprocess =
    let wave := interaction (source_position source_wave) in
    let output :=
      if use_coherence then coherence wave
      else if use_detector then detector wave
      else wave_to_output wave in
    if use_postprocess then postprocess output else output.
Proof.
  intros use_coherence use_detector use_postprocess.
  destruct use_coherence, use_detector, use_postprocess; reflexivity.
Qed.

Lemma hrtem_forward_source_tilt_then_position :
  forall use_coherence use_detector use_postprocess,
    hrtem_forward true true use_coherence use_detector use_postprocess =
    let wave := interaction (source_position (source_tilt source_wave)) in
    let output :=
      if use_coherence then coherence wave
      else if use_detector then detector wave
      else wave_to_output wave in
    if use_postprocess then postprocess output else output.
Proof.
  intros use_coherence use_detector use_postprocess.
  destruct use_coherence, use_detector, use_postprocess; reflexivity.
Qed.

Lemma hrtem_forward_plain_wave_with_optional_source_position :
  forall use_source_tilt use_source_position,
    hrtem_forward use_source_tilt use_source_position false false false =
    wave_to_output
      (interaction
         (if use_source_position
          then source_position
                 (if use_source_tilt then source_tilt source_wave else source_wave)
          else if use_source_tilt then source_tilt source_wave else source_wave)).
Proof.
  intros use_source_tilt use_source_position.
  destruct use_source_tilt, use_source_position; reflexivity.
Qed.

Lemma hrtem_forward_postprocessed_plain_wave :
  hrtem_forward false false false false true =
  postprocess (wave_to_output (interaction source_wave)).
Proof.
  reflexivity.
Qed.

Lemma hrtem_forward_postprocessed_plain_wave_with_optional_source_position :
  forall use_source_tilt use_source_position,
    hrtem_forward use_source_tilt use_source_position false false true =
    postprocess
      (wave_to_output
         (interaction
            (if use_source_position
             then source_position
                    (if use_source_tilt then source_tilt source_wave else source_wave)
             else if use_source_tilt then source_tilt source_wave else source_wave))).
Proof.
  intros use_source_tilt use_source_position.
  destruct use_source_tilt, use_source_position; reflexivity.
Qed.

Lemma hrtem_forward_detector_only :
  forall use_source_tilt use_source_position,
    hrtem_forward use_source_tilt use_source_position false true false =
    detector
      (interaction
         (if use_source_position
          then source_position
                 (if use_source_tilt then source_tilt source_wave else source_wave)
          else if use_source_tilt then source_tilt source_wave else source_wave)).
Proof.
  intros use_source_tilt use_source_position.
  destruct use_source_tilt, use_source_position; reflexivity.
Qed.

Lemma hrtem_forward_detector_only_with_optional_source_position :
  forall use_source_tilt use_source_position,
    hrtem_forward use_source_tilt use_source_position false true false =
    detector
      (interaction
         (if use_source_position
          then source_position
                 (if use_source_tilt then source_tilt source_wave else source_wave)
          else if use_source_tilt then source_tilt source_wave else source_wave)).
Proof.
  intros use_source_tilt use_source_position.
  destruct use_source_tilt, use_source_position; reflexivity.
Qed.

Lemma hrtem_forward_postprocessed_detector_only :
  forall use_source_tilt use_source_position,
    hrtem_forward use_source_tilt use_source_position false true true =
    postprocess
      (detector
         (interaction
            (if use_source_position
             then source_position
                    (if use_source_tilt then source_tilt source_wave else source_wave)
             else if use_source_tilt then source_tilt source_wave else source_wave))).
Proof.
  intros use_source_tilt use_source_position.
  destruct use_source_tilt, use_source_position; reflexivity.
Qed.

Lemma hrtem_forward_postprocessed_detector_only_without_source_tilt_or_position :
  hrtem_forward false false false true true =
  postprocess
    (detector
       (interaction source_wave)).
Proof.
  reflexivity.
Qed.

Lemma hrtem_forward_detector_only_without_source_tilt_or_position :
  hrtem_forward false false false true false =
  detector
    (interaction source_wave).
Proof.
  reflexivity.
Qed.

Lemma hrtem_forward_coherence_precedes_detector_choice :
  forall use_source_tilt use_source_position use_detector,
    hrtem_forward use_source_tilt use_source_position true use_detector false =
    coherence
      (interaction
         (if use_source_position
          then source_position
                 (if use_source_tilt then source_tilt source_wave else source_wave)
          else if use_source_tilt then source_tilt source_wave else source_wave)).
Proof.
  intros use_source_tilt use_source_position use_detector.
  destruct use_source_tilt, use_source_position, use_detector; reflexivity.
Qed.

Lemma hrtem_forward_coherence_ignores_detector_choice :
  forall use_source_tilt use_source_position use_detector1 use_detector2,
    hrtem_forward use_source_tilt use_source_position true use_detector1 false =
    hrtem_forward use_source_tilt use_source_position true use_detector2 false.
Proof.
  intros use_source_tilt use_source_position use_detector1 use_detector2.
  destruct use_source_tilt, use_source_position, use_detector1, use_detector2; reflexivity.
Qed.

Lemma hrtem_forward_postprocess_wraps_output :
  forall use_source_tilt use_source_position use_coherence use_detector,
    hrtem_forward use_source_tilt use_source_position use_coherence use_detector true =
    postprocess
      (hrtem_forward use_source_tilt use_source_position use_coherence use_detector false).
Proof.
  intros use_source_tilt use_source_position use_coherence use_detector.
  destruct use_source_tilt, use_source_position, use_coherence, use_detector; reflexivity.
Qed.

Lemma hrtem_forward_postprocessed_coherence_ignores_detector_choice :
  forall use_source_tilt use_source_position use_detector1 use_detector2,
    hrtem_forward use_source_tilt use_source_position true use_detector1 true =
    hrtem_forward use_source_tilt use_source_position true use_detector2 true.
Proof.
  intros use_source_tilt use_source_position use_detector1 use_detector2.
  destruct use_source_tilt, use_source_position, use_detector1, use_detector2; reflexivity.
Qed.

Lemma hrtem_forward_coherence_without_source_tilt_or_position :
  forall use_detector,
    hrtem_forward false false true use_detector false =
    coherence
      (interaction source_wave).
Proof.
  intro use_detector.
  destruct use_detector; reflexivity.
Qed.

Lemma hrtem_forward_postprocessed_coherence :
  forall use_source_tilt use_source_position use_detector,
    hrtem_forward use_source_tilt use_source_position true use_detector true =
    postprocess
      (coherence
         (interaction
            (if use_source_position
             then source_position
                    (if use_source_tilt then source_tilt source_wave else source_wave)
             else if use_source_tilt then source_tilt source_wave else source_wave))).
Proof.
  intros use_source_tilt use_source_position use_detector.
  destruct use_source_tilt, use_source_position, use_detector; reflexivity.
Qed.

Lemma hrtem_forward_postprocessed_coherence_without_source_tilt_or_position :
  forall use_detector,
    hrtem_forward false false true use_detector true =
    postprocess
      (coherence
         (interaction source_wave)).
Proof.
  intro use_detector.
  destruct use_detector; reflexivity.
Qed.

Lemma stem_forward_plain_waves :
  stem_forward false false false false =
  waves_to_output (map interaction (scan source_wave)).
Proof.
  reflexivity.
Qed.

Lemma stem_forward_without_source_tilt :
  forall use_coherence use_detector use_postprocess,
    stem_forward false use_coherence use_detector use_postprocess =
    let waves := map interaction (scan source_wave) in
    let output :=
      if use_coherence then stack_outputs (map coherence waves)
      else if use_detector then stack_outputs (map detector waves)
      else waves_to_output waves in
    if use_postprocess then postprocess output else output.
Proof.
  intros use_coherence use_detector use_postprocess.
  destruct use_coherence, use_detector, use_postprocess; reflexivity.
Qed.

Lemma stem_forward_with_source_tilt :
  forall use_coherence use_detector use_postprocess,
    stem_forward true use_coherence use_detector use_postprocess =
    let waves := map interaction (scan (source_tilt source_wave)) in
    let output :=
      if use_coherence then stack_outputs (map coherence waves)
      else if use_detector then stack_outputs (map detector waves)
      else waves_to_output waves in
    if use_postprocess then postprocess output else output.
Proof.
  intros use_coherence use_detector use_postprocess.
  destruct use_coherence, use_detector, use_postprocess; reflexivity.
Qed.

Lemma stem_forward_plain_waves_with_optional_source_tilt :
  forall use_source_tilt,
    stem_forward use_source_tilt false false false =
    waves_to_output
      (map interaction
         (scan (if use_source_tilt then source_tilt source_wave else source_wave))).
Proof.
  intro use_source_tilt.
  destruct use_source_tilt; reflexivity.
Qed.

Lemma stem_forward_postprocessed_plain_waves :
  stem_forward false false false true =
  postprocess (waves_to_output (map interaction (scan source_wave))).
Proof.
  reflexivity.
Qed.

Lemma stem_forward_postprocessed_plain_waves_with_optional_source_tilt :
  forall use_source_tilt,
    stem_forward use_source_tilt false false true =
    postprocess
      (waves_to_output
         (map interaction
            (scan (if use_source_tilt then source_tilt source_wave else source_wave)))).
Proof.
  intro use_source_tilt.
  destruct use_source_tilt; reflexivity.
Qed.

Lemma stem_forward_detector_only :
  forall use_source_tilt,
    stem_forward use_source_tilt false true false =
    stack_outputs
      (map detector
        (map interaction
          (scan (if use_source_tilt then source_tilt source_wave else source_wave)))).
Proof.
  intro use_source_tilt.
  destruct use_source_tilt; reflexivity.
Qed.

Lemma stem_forward_detector_only_with_optional_source_tilt :
  forall use_source_tilt,
    stem_forward use_source_tilt false true false =
    stack_outputs
      (map detector
        (map interaction
          (scan (if use_source_tilt then source_tilt source_wave else source_wave)))).
Proof.
  intro use_source_tilt.
  destruct use_source_tilt; reflexivity.
Qed.

Lemma stem_forward_postprocessed_detector_only :
  forall use_source_tilt,
    stem_forward use_source_tilt false true true =
    postprocess
      (stack_outputs
         (map detector
           (map interaction
             (scan (if use_source_tilt then source_tilt source_wave else source_wave))))).
Proof.
  intro use_source_tilt.
  destruct use_source_tilt; reflexivity.
Qed.

Lemma stem_forward_postprocessed_detector_only_without_source_tilt :
  stem_forward false false true true =
  postprocess
    (stack_outputs
       (map detector
          (map interaction (scan source_wave)))).
Proof.
  reflexivity.
Qed.

Lemma stem_forward_coherence_only :
  forall use_source_tilt use_detector,
    stem_forward use_source_tilt true use_detector false =
    stack_outputs
      (map coherence
        (map interaction
          (scan (if use_source_tilt then source_tilt source_wave else source_wave)))).
Proof.
  intros use_source_tilt use_detector.
  destruct use_source_tilt, use_detector; reflexivity.
Qed.

Lemma stem_forward_coherence_only_without_source_tilt :
  forall use_detector,
    stem_forward false true use_detector false =
    stack_outputs
      (map coherence
         (map interaction (scan source_wave))).
Proof.
  intro use_detector.
  destruct use_detector; reflexivity.
Qed.

Lemma stem_forward_coherence_ignores_detector_choice :
  forall use_source_tilt use_detector1 use_detector2,
    stem_forward use_source_tilt true use_detector1 false =
    stem_forward use_source_tilt true use_detector2 false.
Proof.
  intros use_source_tilt use_detector1 use_detector2.
  destruct use_source_tilt, use_detector1, use_detector2; reflexivity.
Qed.

Lemma stem_forward_postprocess_wraps_output :
  forall use_source_tilt use_coherence use_detector,
    stem_forward use_source_tilt use_coherence use_detector true =
    postprocess (stem_forward use_source_tilt use_coherence use_detector false).
Proof.
  intros use_source_tilt use_coherence use_detector.
  destruct use_source_tilt, use_coherence, use_detector; reflexivity.
Qed.

Lemma stem_forward_postprocessed_coherence_ignores_detector_choice :
  forall use_source_tilt use_detector1 use_detector2,
    stem_forward use_source_tilt true use_detector1 true =
    stem_forward use_source_tilt true use_detector2 true.
Proof.
  intros use_source_tilt use_detector1 use_detector2.
  destruct use_source_tilt, use_detector1, use_detector2; reflexivity.
Qed.

Lemma stem_forward_postprocessed_coherence :
  forall use_source_tilt use_detector,
    stem_forward use_source_tilt true use_detector true =
    postprocess
      (stack_outputs
         (map coherence
            (map interaction
               (scan (if use_source_tilt then source_tilt source_wave else source_wave))))).
Proof.
  intros use_source_tilt use_detector.
  destruct use_source_tilt, use_detector; reflexivity.
Qed.

Lemma stem_forward_postprocessed_coherence_without_source_tilt :
  forall use_detector,
    stem_forward false true use_detector true =
    postprocess
      (stack_outputs
         (map coherence
            (map interaction (scan source_wave)))).
Proof.
  intro use_detector.
  destruct use_detector; reflexivity.
Qed.

End ForwardEquationCompositionLaws.
