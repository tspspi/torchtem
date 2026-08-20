Require Import List.
Require Import Reals.

Import ListNotations.

Section CoherenceReadoutLaws.

Variable Wave DetectorOutput : Type.

Variable apply_source_offsets : Wave -> list Wave.
Variable average_intensity : list Wave -> DetectorOutput.
Variable apply_detector : nat -> Wave -> DetectorOutput.
Variable stack_outputs : list DetectorOutput -> DetectorOutput.
Variable named_detector_ids : list nat.

Definition apply_single_unnamed (wave : Wave) : DetectorOutput :=
  average_intensity (apply_source_offsets wave).

Definition apply_single_named (wave : Wave) : list (nat * DetectorOutput) :=
  map
    (fun detector_id =>
       (detector_id, average_intensity (apply_source_offsets wave)))
    named_detector_ids.

Definition coherence_forward_unnamed (waves : list Wave) : DetectorOutput :=
  match waves with
  | [wave] => apply_single_unnamed wave
  | _ => stack_outputs (map apply_single_unnamed waves)
  end.

Definition coherence_forward_named
  (waves : list Wave) : list (nat * list DetectorOutput) :=
  map
    (fun detector_id =>
       (detector_id,
        map
          (fun wave => average_intensity (apply_source_offsets wave))
          waves))
    named_detector_ids.

Lemma coherence_forward_unnamed_singleton :
  forall wave,
    coherence_forward_unnamed [wave] =
    average_intensity (apply_source_offsets wave).
Proof.
  reflexivity.
Qed.

Lemma coherence_forward_unnamed_batch :
  forall waves,
    waves <> [] ->
    (forall wave, waves <> [wave]) ->
    coherence_forward_unnamed waves =
    stack_outputs
      (map (fun wave => average_intensity (apply_source_offsets wave)) waves).
Proof.
  intros waves Hnonempty Hnot_singleton.
  destruct waves as [|wave waves].
  - exfalso. apply Hnonempty. reflexivity.
  - destruct waves as [|wave' rest].
    + exfalso. apply (Hnot_singleton wave). reflexivity.
    + reflexivity.
Qed.

Lemma coherence_forward_named_detector_count :
  forall waves,
    length (coherence_forward_named waves) = length named_detector_ids.
Proof.
  intro waves.
  unfold coherence_forward_named.
  rewrite map_length.
  reflexivity.
Qed.

Lemma coherence_forward_named_outputs_follow_input_length :
  forall waves detector_id outputs,
    In (detector_id, outputs) (coherence_forward_named waves) ->
    length outputs = length waves.
Proof.
  intros waves detector_id outputs Hin.
  unfold coherence_forward_named in Hin.
  apply in_map_iff in Hin.
  destruct Hin as [source [Hpair Hin_ids]].
  inversion Hpair; subst.
  rewrite map_length.
  reflexivity.
Qed.

End CoherenceReadoutLaws.

Section ModeReductionLaws.

Variable Output : Type.

Variable sum_outputs : list Output -> Output.
Variable weighted_sum_outputs : list R -> list Output -> Output.
Variable apply_detector_recursive : nat -> Output -> Output.
Variable named_detector_ids : list nat.

Definition reduce_outputs_unweighted (outputs : list Output) : Output :=
  sum_outputs outputs.

Definition reduce_outputs_weighted (weights : list R) (outputs : list Output) : Output :=
  weighted_sum_outputs weights outputs.

Definition mode_forward_unnamed
  (detector_id : option nat)
  (wave_intensities : list Output)
  (weights : option (list R)) : Output :=
  let outputs :=
    match detector_id with
    | None => wave_intensities
    | Some detector => map (apply_detector_recursive detector) wave_intensities
    end in
  match weights with
  | None => reduce_outputs_unweighted outputs
  | Some ws => reduce_outputs_weighted ws outputs
  end.

Definition mode_forward_named
  (wave_intensities : list Output)
  (weights : option (list R)) : list (nat * Output) :=
  map
    (fun detector_id =>
       (detector_id,
        match weights with
        | None =>
            reduce_outputs_unweighted
              (map (apply_detector_recursive detector_id) wave_intensities)
        | Some ws =>
            reduce_outputs_weighted ws
              (map (apply_detector_recursive detector_id) wave_intensities)
        end))
    named_detector_ids.

Lemma mode_forward_unnamed_without_detector_unweighted :
  forall outputs,
    mode_forward_unnamed None outputs None = sum_outputs outputs.
Proof.
  reflexivity.
Qed.

Lemma mode_forward_unnamed_with_detector_weighted :
  forall detector_id weights outputs,
    mode_forward_unnamed (Some detector_id) outputs (Some weights) =
    weighted_sum_outputs weights (map (apply_detector_recursive detector_id) outputs).
Proof.
  reflexivity.
Qed.

Lemma mode_forward_named_detector_count :
  forall outputs weights,
    length (mode_forward_named outputs weights) = length named_detector_ids.
Proof.
  intros outputs weights.
  unfold mode_forward_named.
  rewrite map_length.
  reflexivity.
Qed.

End ModeReductionLaws.
