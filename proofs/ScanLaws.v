Require Import Reals.
Require Import Psatz.

Open Scope R_scope.

Definition affine_interp (start_A end_A t : R) : R :=
  start_A + (end_A - start_A) * t.

Lemma affine_interp_zero :
  forall start_A end_A : R,
    affine_interp start_A end_A 0 = start_A.
Proof.
  intros.
  unfold affine_interp.
  lra.
Qed.

Lemma affine_interp_one :
  forall start_A end_A : R,
    affine_interp start_A end_A 1 = end_A.
Proof.
  intros.
  unfold affine_interp.
  lra.
Qed.

Lemma affine_interp_half :
  forall start_A end_A : R,
    affine_interp start_A end_A (/ 2) = (start_A + end_A) / 2.
Proof.
  intros.
  unfold affine_interp.
  field.
Qed.

Lemma affine_interp_reverse_parameter :
  forall start_A end_A t : R,
    affine_interp start_A end_A t =
    affine_interp end_A start_A (1 - t).
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_convex_combination :
  forall start_A end_A t : R,
    affine_interp start_A end_A t =
    (1 - t) * start_A + t * end_A.
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_difference :
  forall start_A end_A t1 t2 : R,
    affine_interp start_A end_A t2 - affine_interp start_A end_A t1 =
    (end_A - start_A) * (t2 - t1).
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_translation :
  forall start_A end_A t delta : R,
    affine_interp (start_A + delta) (end_A + delta) t =
    affine_interp start_A end_A t + delta.
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_complementary_sum :
  forall start_A end_A t : R,
    affine_interp start_A end_A t +
    affine_interp start_A end_A (1 - t) =
    start_A + end_A.
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_average_parameters :
  forall start_A end_A t1 t2 : R,
    affine_interp start_A end_A ((t1 + t2) / 2) =
    (affine_interp start_A end_A t1 + affine_interp start_A end_A t2) / 2.
Proof.
  intros.
  unfold affine_interp.
  field.
Qed.

Lemma affine_interp_scaling :
  forall start_A end_A t scale : R,
    affine_interp (scale * start_A) (scale * end_A) t =
    scale * affine_interp start_A end_A t.
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_idempotent :
  forall start_A end_A t s : R,
    affine_interp
      (affine_interp start_A end_A t)
      (affine_interp start_A end_A t)
      s =
    affine_interp start_A end_A t.
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_nested :
  forall start_A end_A t1 t2 s : R,
    affine_interp
      (affine_interp start_A end_A t1)
      (affine_interp start_A end_A t2)
      s =
    affine_interp start_A end_A (affine_interp t1 t2 s).
Proof.
  intros.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_subsegment :
  forall start_A end_A t1 t2 s : R,
    affine_interp
      (affine_interp start_A end_A t1)
      (affine_interp start_A end_A t2)
      s =
    affine_interp start_A end_A ((1 - s) * t1 + s * t2).
Proof.
  intros.
  rewrite affine_interp_nested.
  rewrite affine_interp_convex_combination.
  unfold affine_interp.
  ring.
Qed.

Lemma affine_interp_constant :
  forall start_A t : R,
    affine_interp start_A start_A t = start_A.
Proof.
  intros.
  unfold affine_interp.
  lra.
Qed.

Lemma affine_interp_within_bounds :
  forall start_A end_A t : R,
    start_A <= end_A ->
    0 <= t <= 1 ->
    start_A <= affine_interp start_A end_A t <= end_A.
Proof.
  intros start_A end_A t Horder Ht.
  unfold affine_interp.
  nra.
Qed.

Lemma affine_interp_within_reversed_bounds :
  forall start_A end_A t : R,
    end_A <= start_A ->
    0 <= t <= 1 ->
    end_A <= affine_interp start_A end_A t <= start_A.
Proof.
  intros start_A end_A t Horder Ht.
  unfold affine_interp.
  nra.
Qed.

Lemma affine_interp_monotone :
  forall start_A end_A t1 t2 : R,
    start_A <= end_A ->
    t1 <= t2 ->
    affine_interp start_A end_A t1 <= affine_interp start_A end_A t2.
Proof.
  intros start_A end_A t1 t2 Horder Ht.
  unfold affine_interp.
  nra.
Qed.

Lemma affine_interp_monotone_reversed :
  forall start_A end_A t1 t2 : R,
    end_A <= start_A ->
    t1 <= t2 ->
    affine_interp start_A end_A t2 <= affine_interp start_A end_A t1.
Proof.
  intros start_A end_A t1 t2 Horder Ht.
  unfold affine_interp.
  nra.
Qed.

Lemma affine_interp_strictly_monotone :
  forall start_A end_A t1 t2 : R,
    start_A < end_A ->
    t1 < t2 ->
    affine_interp start_A end_A t1 < affine_interp start_A end_A t2.
Proof.
  intros start_A end_A t1 t2 Horder Ht.
  unfold affine_interp.
  nra.
Qed.

Lemma affine_interp_strictly_monotone_reversed :
  forall start_A end_A t1 t2 : R,
    end_A < start_A ->
    t1 < t2 ->
    affine_interp start_A end_A t2 < affine_interp start_A end_A t1.
Proof.
  intros start_A end_A t1 t2 Horder Ht.
  unfold affine_interp.
  nra.
Qed.
