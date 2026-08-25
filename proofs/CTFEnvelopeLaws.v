Require Import Reals.
Require Import Psatz.

Open Scope R_scope.

Definition temporal_envelope
  (wavelength focal_spread alpha : R) : R :=
  exp (- ((0.5 * PI / wavelength * focal_spread * alpha ^ 2) ^ 2)).

Definition spatial_envelope
  (angular_spread_sign angular_spread dchi_dk dchi_dphi : R) : R :=
  exp
    (- angular_spread_sign
       * (angular_spread / 2) ^ 2
       * (dchi_dk ^ 2 + dchi_dphi ^ 2)).

Lemma temporal_envelope_zero_focal_spread :
  forall wavelength alpha,
    temporal_envelope wavelength 0 alpha = 1.
Proof.
  intros wavelength alpha.
  unfold temporal_envelope.
  replace (- ((0.5 * PI / wavelength * 0 * alpha ^ 2) ^ 2)) with 0 by nra.
  apply exp_0.
Qed.

Lemma temporal_envelope_zero_angle :
  forall wavelength focal_spread,
    temporal_envelope wavelength focal_spread 0 = 1.
Proof.
  intros wavelength focal_spread.
  unfold temporal_envelope.
  replace (- ((0.5 * PI / wavelength * focal_spread * 0 ^ 2) ^ 2)) with 0 by nra.
  apply exp_0.
Qed.

Lemma spatial_envelope_zero_angular_spread :
  forall angular_spread_sign dchi_dk dchi_dphi,
    spatial_envelope angular_spread_sign 0 dchi_dk dchi_dphi = 1.
Proof.
  intros angular_spread_sign dchi_dk dchi_dphi.
  unfold spatial_envelope.
  replace
    (- angular_spread_sign
       * (0 / 2) ^ 2
       * (dchi_dk ^ 2 + dchi_dphi ^ 2))
    with 0 by nra.
  apply exp_0.
Qed.

Lemma spatial_envelope_zero_phase_gradients :
  forall angular_spread_sign angular_spread,
    spatial_envelope angular_spread_sign angular_spread 0 0 = 1.
Proof.
  intros angular_spread_sign angular_spread.
  unfold spatial_envelope.
  replace
    (- angular_spread_sign
       * (angular_spread / 2) ^ 2
       * (0 ^ 2 + 0 ^ 2))
    with 0 by nra.
  apply exp_0.
Qed.

Section CTFForwardAdjustmentLaws.

Variable Transfer : Type.

Variable apply_spatial_envelope : Transfer -> Transfer.
Variable apply_temporal_envelope : Transfer -> Transfer.
Variable apply_wiener_filter : Transfer -> Transfer.
Variable apply_flip_phase : Transfer -> Transfer.

Definition ctf_forward_adjustment
  (use_spatial_envelope use_temporal_envelope use_wiener_filter flip_phase : bool)
  (base_transfer : Transfer) : Transfer :=
  let transfer1 :=
    if use_spatial_envelope
    then apply_spatial_envelope base_transfer
    else base_transfer in
  let transfer2 :=
    if use_temporal_envelope
    then apply_temporal_envelope transfer1
    else transfer1 in
  if use_wiener_filter
  then apply_wiener_filter transfer2
  else if flip_phase then apply_flip_phase transfer2 else transfer2.

Lemma ctf_forward_adjustment_identity :
  forall base_transfer,
    ctf_forward_adjustment false false false false base_transfer = base_transfer.
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_spatial_only :
  forall base_transfer,
    ctf_forward_adjustment true false false false base_transfer =
    apply_spatial_envelope base_transfer.
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_temporal_only :
  forall base_transfer,
    ctf_forward_adjustment false true false false base_transfer =
    apply_temporal_envelope base_transfer.
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_flip_only :
  forall base_transfer,
    ctf_forward_adjustment false false false true base_transfer =
    apply_flip_phase base_transfer.
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_wiener_only :
  forall flip_phase base_transfer,
    ctf_forward_adjustment false false true flip_phase base_transfer =
    apply_wiener_filter base_transfer.
Proof.
  intros flip_phase base_transfer.
  destruct flip_phase; reflexivity.
Qed.

Lemma ctf_forward_adjustment_temporal_then_flip_phase :
  forall base_transfer,
    ctf_forward_adjustment false true false true base_transfer =
    apply_flip_phase (apply_temporal_envelope base_transfer).
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_spatial_then_flip_phase :
  forall base_transfer,
    ctf_forward_adjustment true false false true base_transfer =
    apply_flip_phase (apply_spatial_envelope base_transfer).
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_spatial_then_wiener :
  forall flip_phase base_transfer,
    ctf_forward_adjustment true false true flip_phase base_transfer =
    apply_wiener_filter (apply_spatial_envelope base_transfer).
Proof.
  intros flip_phase base_transfer.
  destruct flip_phase; reflexivity.
Qed.

Lemma ctf_forward_adjustment_temporal_then_wiener :
  forall flip_phase base_transfer,
    ctf_forward_adjustment false true true flip_phase base_transfer =
    apply_wiener_filter (apply_temporal_envelope base_transfer).
Proof.
  intros flip_phase base_transfer.
  destruct flip_phase; reflexivity.
Qed.

Lemma ctf_forward_adjustment_spatial_then_temporal :
  forall base_transfer,
    ctf_forward_adjustment true true false false base_transfer =
    apply_temporal_envelope (apply_spatial_envelope base_transfer).
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_spatial_then_temporal_then_flip_phase :
  forall base_transfer,
    ctf_forward_adjustment true true false true base_transfer =
    apply_flip_phase
      (apply_temporal_envelope (apply_spatial_envelope base_transfer)).
Proof.
  reflexivity.
Qed.

Lemma ctf_forward_adjustment_spatial_then_temporal_then_wiener :
  forall flip_phase base_transfer,
    ctf_forward_adjustment true true true flip_phase base_transfer =
    apply_wiener_filter
      (apply_temporal_envelope (apply_spatial_envelope base_transfer)).
Proof.
  intros flip_phase base_transfer.
  destruct flip_phase; reflexivity.
Qed.

Lemma ctf_forward_adjustment_spatial_as_base_wrapper :
  forall use_temporal_envelope use_wiener_filter flip_phase base_transfer,
    ctf_forward_adjustment
      true use_temporal_envelope use_wiener_filter flip_phase base_transfer =
    ctf_forward_adjustment
      false use_temporal_envelope use_wiener_filter flip_phase
      (apply_spatial_envelope base_transfer).
Proof.
  intros use_temporal_envelope use_wiener_filter flip_phase base_transfer.
  destruct use_temporal_envelope, use_wiener_filter, flip_phase; reflexivity.
Qed.

Lemma ctf_forward_adjustment_temporal_as_base_wrapper :
  forall use_spatial_envelope use_wiener_filter flip_phase base_transfer,
    ctf_forward_adjustment
      use_spatial_envelope true use_wiener_filter flip_phase base_transfer =
    ctf_forward_adjustment
      false false use_wiener_filter flip_phase
      (apply_temporal_envelope
         (if use_spatial_envelope
          then apply_spatial_envelope base_transfer
          else base_transfer)).
Proof.
  intros use_spatial_envelope use_wiener_filter flip_phase base_transfer.
  destruct use_spatial_envelope, use_wiener_filter, flip_phase; reflexivity.
Qed.

Lemma ctf_forward_adjustment_flip_phase_without_wiener :
  forall use_spatial_envelope use_temporal_envelope base_transfer,
    ctf_forward_adjustment
      use_spatial_envelope use_temporal_envelope false true base_transfer =
    apply_flip_phase
      (ctf_forward_adjustment
         use_spatial_envelope use_temporal_envelope false false base_transfer).
Proof.
  intros use_spatial_envelope use_temporal_envelope base_transfer.
  destruct use_spatial_envelope, use_temporal_envelope; reflexivity.
Qed.

Lemma ctf_forward_adjustment_wiener_precedes_flip_phase :
  forall use_spatial_envelope use_temporal_envelope flip_phase base_transfer,
    ctf_forward_adjustment
      use_spatial_envelope use_temporal_envelope true flip_phase base_transfer =
    apply_wiener_filter
      (ctf_forward_adjustment
         use_spatial_envelope use_temporal_envelope false false base_transfer).
Proof.
  intros use_spatial_envelope use_temporal_envelope flip_phase base_transfer.
  destruct use_spatial_envelope, use_temporal_envelope, flip_phase; reflexivity.
Qed.

Lemma ctf_forward_adjustment_wiener_makes_flip_phase_irrelevant :
  forall use_spatial_envelope use_temporal_envelope base_transfer,
    ctf_forward_adjustment
      use_spatial_envelope use_temporal_envelope true true base_transfer =
    ctf_forward_adjustment
      use_spatial_envelope use_temporal_envelope true false base_transfer.
Proof.
  intros use_spatial_envelope use_temporal_envelope base_transfer.
  destruct use_spatial_envelope, use_temporal_envelope; reflexivity.
Qed.

End CTFForwardAdjustmentLaws.
