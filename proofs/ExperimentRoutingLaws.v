Require Import List.
Require Import String.

Import ListNotations.

Inductive backend :=
| MultisliceBackend
| BlochBackend
| PrismBackend.

Inductive detector_kind :=
| PixelatedDetector
| ImageDetector
| OtherDetector.

Inductive source_kind :=
| ProbeSource
| PlaneWaveSource.

Inductive mode :=
| InelasticMode
| MagneticMode
| DiffractionMode
| STEMMode
| HRTEMMode.

Record experiment_signature := {
  has_inelastic : bool;
  has_magnetic : bool;
  backend_kind : backend;
  scan_present : bool;
  source_kind_value : source_kind;
  detector_present : bool;
  detector_is_pixelated : bool;
  detector_is_image : bool;
  named_detector_kinds : list detector_kind
}.

Definition all_pixelated (kinds : list detector_kind) : bool :=
  forallb
    (fun kind =>
       match kind with
       | PixelatedDetector => true
       | _ => false
       end)
    kinds.

Definition all_image (kinds : list detector_kind) : bool :=
  forallb
    (fun kind =>
       match kind with
       | ImageDetector => true
       | _ => false
       end)
    kinds.

Definition infer_mode_model (config : experiment_signature) : mode :=
  if has_inelastic config then InelasticMode
  else if has_magnetic config then MagneticMode
  else match backend_kind config with
       | BlochBackend => DiffractionMode
       | _ =>
           if detector_present config then
             if detector_is_pixelated config then
               if scan_present config then DiffractionMode else HRTEMMode
             else if detector_is_image config then
               HRTEMMode
             else if all_pixelated (named_detector_kinds config) then
               if scan_present config then DiffractionMode else HRTEMMode
             else if all_image (named_detector_kinds config) then
               HRTEMMode
             else if andb (scan_present config)
                           match source_kind_value config with
                           | ProbeSource => true
                           | PlaneWaveSource => false
                           end
                  then STEMMode
                  else if scan_present config then STEMMode else HRTEMMode
           else if andb (scan_present config)
                         match source_kind_value config with
                         | ProbeSource => true
                         | PlaneWaveSource => false
                         end
                then STEMMode
                else if scan_present config then STEMMode else HRTEMMode
       end.

Definition detector_names_model
  (detector_names : list string)
  (detector_present : bool)
  (named_detector_map : bool) : list string :=
  if detector_present then
    if named_detector_map then detector_names else ["detector"%string]
  else ["exit_wave"%string].

Lemma infer_mode_model_inelastic :
  forall config,
    has_inelastic config = true ->
    infer_mode_model config = InelasticMode.
Proof.
  intros config H.
  unfold infer_mode_model.
  rewrite H.
  reflexivity.
Qed.

Lemma infer_mode_model_magnetic :
  forall config,
    has_inelastic config = false ->
    has_magnetic config = true ->
    infer_mode_model config = MagneticMode.
Proof.
  intros config Hinel Hmag.
  unfold infer_mode_model.
  rewrite Hinel, Hmag.
  reflexivity.
Qed.

Lemma infer_mode_model_bloch :
  forall config,
    has_inelastic config = false ->
    has_magnetic config = false ->
    backend_kind config = BlochBackend ->
    infer_mode_model config = DiffractionMode.
Proof.
  intros config Hinel Hmag Hbackend.
  unfold infer_mode_model.
  rewrite Hinel, Hmag, Hbackend.
  reflexivity.
Qed.

Lemma detector_names_model_without_detector :
  forall names named_detector_map,
    detector_names_model names false named_detector_map = ["exit_wave"%string].
Proof.
  reflexivity.
Qed.

Lemma detector_names_model_with_single_detector :
  forall names,
    detector_names_model names true false = ["detector"%string].
Proof.
  reflexivity.
Qed.

Lemma detector_names_model_with_named_detectors :
  forall names,
    detector_names_model names true true = names.
Proof.
  reflexivity.
Qed.
